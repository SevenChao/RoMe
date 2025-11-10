#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive mask creation tool using OpenCV
Allows user to manually click or draw polygons to create masks for images
"""

import cv2
import numpy as np
import os
import argparse


class MaskCreator:
    def __init__(self, image_path, output_path=None, max_display_size=(1920, 1080)):
        """
        Initialize mask creator
        
        Args:
            image_path: Path to input image
            output_path: Path to save mask (optional, defaults to image_path with _mask suffix)
            max_display_size: Maximum display size (width, height) for scaling large images
        """
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Store original dimensions
        self.original_height, self.original_width = self.original_image.shape[:2]
        
        # Calculate scale factor to fit within max_display_size
        scale_w = max_display_size[0] / self.original_width if self.original_width > max_display_size[0] else 1.0
        scale_h = max_display_size[1] / self.original_height if self.original_height > max_display_size[1] else 1.0
        self.scale_factor = min(scale_w, scale_h, 1.0)  # Never upscale, only downscale
        
        # Scale image for display
        if self.scale_factor < 1.0:
            display_width = int(self.original_width * self.scale_factor)
            display_height = int(self.original_height * self.scale_factor)
            self.image = cv2.resize(self.original_image, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
            print(f"Image scaled from {self.original_width}x{self.original_height} to {display_width}x{display_height} for display")
        else:
            self.image = self.original_image.copy()
            print(f"Image size: {self.original_width}x{self.original_height} (no scaling needed)")
        
        # Set output path - save to current working directory by default
        if output_path is None:
            image_basename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(image_basename)[0]
            # Save to current working directory
            output_path = os.path.join(os.getcwd(), f"{name_without_ext}_mask.png")  # Always save as PNG
        else:
            # If output path is relative, make it relative to current directory
            if not os.path.isabs(output_path):
                output_path = os.path.join(os.getcwd(), output_path)
        self.output_path = output_path
        
        # Create mask (initially all zeros - nothing filtered) - use display size
        # User will mark areas to FILTER OUT (set to 255)
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        
        # Drawing state
        self.drawing = False
        self.mode = 'polygon'  # 'polygon' or 'brush'
        self.points = []  # For polygon mode
        self.current_point = None
        
        # Brush settings
        self.brush_size = 20
        
        # Display image with overlay
        self.display_image = self.image.copy()
        self.window_name = "Mask Creator - Press 'h' for help"
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        # Validate coordinates
        if x is None or y is None:
            return
        
        # Clamp coordinates to image bounds
        x = max(0, min(x, self.image.shape[1] - 1))
        y = max(0, min(y, self.image.shape[0] - 1))
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == 'polygon':
                # Add point to polygon
                self.points.append((x, y))
                self.current_point = (x, y)
                # Draw point on display
                try:
                    cv2.circle(self.display_image, (int(x), int(y)), 5, (0, 255, 0), -1)
                    if len(self.points) > 1:
                        cv2.line(self.display_image, self.points[-2], self.points[-1], (0, 255, 0), 2)
                    cv2.imshow(self.window_name, self.display_image)
                except Exception as e:
                    print(f"Warning: Error drawing point: {e}")
            elif self.mode == 'brush':
                self.drawing = True
                try:
                    cv2.circle(self.mask, (int(x), int(y)), self.brush_size, 255, -1)  # 255 = filter out
                    cv2.circle(self.display_image, (int(x), int(y)), self.brush_size, (0, 255, 0), 2)  # Green = filtered
                    cv2.imshow(self.window_name, self.display_image)
                except Exception as e:
                    print(f"Warning: Error drawing brush: {e}")
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.mode == 'brush':
                self.drawing = False
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mode == 'brush' and self.drawing:
                try:
                    cv2.circle(self.mask, (int(x), int(y)), self.brush_size, 255, -1)  # 255 = filter out
                    cv2.circle(self.display_image, (int(x), int(y)), self.brush_size, (0, 255, 0), 2)  # Green = filtered
                    cv2.imshow(self.window_name, self.display_image)
                except Exception as e:
                    print(f"Warning: Error drawing brush: {e}")
            
            if self.mode == 'polygon' and self.current_point:
                # Draw temporary line from last point to current mouse position
                try:
                    temp_image = self.display_image.copy()
                    if len(self.points) > 0:
                        cv2.line(temp_image, self.points[-1], (int(x), int(y)), (255, 255, 0), 2)
                    cv2.imshow(self.window_name, temp_image)
                except Exception as e:
                    pass  # Silently ignore preview errors
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click: finish polygon and mark area as FILTER region
            if self.mode == 'polygon' and len(self.points) >= 3:
                # Close and fill polygon (set to 255 = filter out)
                pts = np.array(self.points, dtype=np.int32)
                cv2.fillPoly(self.mask, [pts], 255)  # 255 = filter out
                # Draw filled polygon on display (green overlay for filtered areas)
                overlay = self.display_image.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))  # Green overlay
                cv2.addWeighted(overlay, 0.3, self.display_image, 0.7, 0, self.display_image)
                self.points = []
                self.current_point = None
                cv2.imshow(self.window_name, self.display_image)
                print(f"Marked area as FILTER region (will be masked out)")
    
    def update_display(self):
        """Update display with current mask overlay"""
        try:
            overlay = self.image.copy()
            if self.mask is None or self.mask.size == 0:
                self.display_image = overlay
                cv2.imshow(self.window_name, self.display_image)
                return
            
            # Apply colormap to mask
            mask_colored = cv2.applyColorMap(self.mask, cv2.COLORMAP_JET)
            mask_binary = self.mask > 0
            
            # Only update where mask exists
            if np.any(mask_binary):
                mask_indices = np.where(mask_binary)
                overlay_pixels = overlay[mask_indices[0], mask_indices[1]]
                colored_pixels = mask_colored[mask_indices[0], mask_indices[1]]
                
                blended = cv2.addWeighted(
                    overlay_pixels.astype(np.float32), 0.5, 
                    colored_pixels.astype(np.float32), 0.5, 
                    0
                ).astype(np.uint8)
                overlay[mask_indices[0], mask_indices[1]] = blended
            
            self.display_image = overlay
            cv2.imshow(self.window_name, self.display_image)
        except Exception as e:
            print(f"Warning: Error updating display: {e}")
            self.display_image = self.image.copy()
            cv2.imshow(self.window_name, self.display_image)
    
    def run(self):
        """Run the interactive mask creator"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Initial display
        self.update_display()
        cv2.imshow(self.window_name, self.display_image)
        
        print("\n" + "="*60)
        print("Mask Creator - Interactive Tool")
        print("="*60)
        print("IMPORTANT: Marked areas will be FILTERED OUT (masked)")
        print("  - White (255) in saved mask = filtered region")
        print("  - Black (0) in saved mask = kept region")
        print("="*60)
        print("Mouse Controls:")
        print("  Left Click: Add point (polygon mode) or draw filter region (brush mode)")
        print("  Right Click: Finish polygon and mark as FILTER region")
        print("  Mouse Move: Draw filter region with brush (brush mode)")
        print("\nKeyboard Controls:")
        print("  'p' - Switch to Polygon mode")
        print("  'b' - Switch to Brush mode")
        print("  '+' / '-' - Increase/Decrease brush size")
        print("  'c' - Clear all filter regions")
        print("  'u' - Undo last polygon")
        print("  's' - Save mask and exit")
        print("  'q' / ESC - Exit without saving")
        print("  'h' - Show this help")
        print("="*60 + "\n")
        
        while True:
            key_code = cv2.waitKey(1)
            if key_code == -1:  # No key pressed
                continue
            key = key_code & 0xFF
            
            if key == ord('q') or key == 27:  # ESC or 'q'
                print("Exiting without saving...")
                break
            
            elif key == ord('s'):
                # Save mask - rescale to original size if needed
                if self.scale_factor < 1.0:
                    # Resize mask to original image size
                    original_mask = cv2.resize(
                        self.mask, 
                        (self.original_width, self.original_height), 
                        interpolation=cv2.INTER_NEAREST
                    )
                    print(f"Mask resized from {self.mask.shape[1]}x{self.mask.shape[0]} to {self.original_width}x{self.original_height}")
                else:
                    original_mask = self.mask
                
                # Save mask: white (255) = filter out, black (0) = keep
                cv2.imwrite(self.output_path, original_mask)
                print(f"Mask saved to: {self.output_path}")
                print(f"Mask size: {original_mask.shape[1]}x{original_mask.shape[0]} (matches original image)")
                filtered_pixels = np.sum(original_mask > 0)
                total_pixels = original_mask.size
                print(f"Filtered regions: {filtered_pixels}/{total_pixels} pixels ({100*filtered_pixels/total_pixels:.2f}%)")
                break
            
            elif key == ord('p'):
                self.mode = 'polygon'
                print(f"Mode switched to: Polygon")
            
            elif key == ord('b'):
                self.mode = 'brush'
                self.points = []
                self.current_point = None
                print(f"Mode switched to: Brush")
            
            elif key == ord('+') or key == ord('='):
                self.brush_size = min(100, self.brush_size + 5)
                print(f"Brush size: {self.brush_size}")
            
            elif key == ord('-') or key == ord('_'):
                self.brush_size = max(5, self.brush_size - 5)
                print(f"Brush size: {self.brush_size}")
            
            elif key == ord('c'):
                self.mask = np.zeros(self.mask.shape, dtype=np.uint8)  # Clear all filter regions
                self.display_image = self.image.copy()
                self.points = []
                self.current_point = None
                self.update_display()
                cv2.imshow(self.window_name, self.display_image)
                print("All filter regions cleared")
            
            elif key == ord('u'):
                # Undo: clear mask and reset points (simple undo)
                if self.points:
                    self.points = []
                    self.current_point = None
                    print("Undone last polygon")
                else:
                    print("Nothing to undo")
            
            elif key == ord('h'):
                print("\n" + "="*60)
                print("Help - Keyboard Controls:")
                print("  'p' - Polygon mode (mark filter regions)")
                print("  'b' - Brush mode (draw filter regions)")
                print("  '+' / '-' - Brush size")
                print("  'c' - Clear all filter regions")
                print("  'u' - Undo last polygon")
                print("  's' - Save mask (white=filter, black=keep)")
                print("  'q' - Quit")
                print("\nNote: Marked areas will be FILTERED OUT (masked)")
                print("="*60 + "\n")
        
        cv2.destroyAllWindows()
        return self.mask


def main():
    parser = argparse.ArgumentParser(description="Interactive mask creation tool for single image")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, default=None, 
                       help="Output mask path (default: image_path_mask.png)")
    parser.add_argument("--max-width", type=int, default=1920,
                       help="Maximum display width (default: 1920)")
    parser.add_argument("--max-height", type=int, default=1080,
                       help="Maximum display height (default: 1080)")
    
    args = parser.parse_args()
    
    # Process single image
    creator = MaskCreator(args.image, args.output, max_display_size=(args.max_width, args.max_height))
    creator.run()


if __name__ == "__main__":
    main()

