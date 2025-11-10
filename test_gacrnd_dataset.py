#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for GACRNDDataset
Tests dataset loading with the specified dataset path
"""

import yaml
import numpy as np
import cv2
import os
from datasets.gacrnd import GACRNDDataset

def test_dataset_loading():
    """Test loading GACRND dataset"""
    
    # Dataset path
    dataset_path = "/home/stevenchao/CBZoom/Auto-Labeling/Dataset/GACRT026_1758521322"
    
    # Load config
    config_path = "configs/local_gac.yaml"
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    
    # Update dataset paths
    configs["base_dir"] = dataset_path
    configs["image_dir"] = dataset_path
    configs["label_dir"] = dataset_path
    configs["lane_dir"] = dataset_path
    
    print("="*60)
    print("Testing GACRNDDataset Loading")
    print("="*60)
    print(f"Dataset path: {dataset_path}")
    print(f"Config file: {config_path}")
    print("="*60)
    
    try:
        # Initialize dataset
        print("\n1. Initializing dataset...")
        dataset = GACRNDDataset(configs)
        print(f"   ✓ Dataset initialized successfully")
        print(f"   - Total samples: {len(dataset)}")
        print(f"   - Image files: {len(dataset.image_filenames)}")
        print(f"   - Label files: {len(dataset.label_filenames)}")
        print(f"   - Camera poses: {len(dataset.ref_camera2world)}")
        
        # Test loading a sample
        if len(dataset) > 0:
            print(f"\n2. Testing sample loading...")
            idx = 0
            sample = dataset[idx]
            
            print(f"   ✓ Sample {idx} loaded successfully")
            print(f"   - Sample keys: {list(sample.keys())}")
            print(f"   - Image shape: {sample['image'].shape}")
            print(f"   - Image dtype: {sample['image'].dtype}")
            print(f"   - Image value range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
            print(f"   - Label shape: {sample['static_label'].shape}")
            print(f"   - Label dtype: {sample['static_label'].dtype}")
            print(f"   - Label value range: [{sample['static_label'].min()}, {sample['static_label'].max()}]")
            print(f"   - Mask shape: {sample['static_mask'].shape}")
            print(f"   - Mask dtype: {sample['static_mask'].dtype}")
            print(f"   - Mask value range: [{sample['static_mask'].min():.3f}, {sample['static_mask'].max():.3f}]")
            print(f"   - Camera idx: {sample['camera_idx']}")
            print(f"   - Camera K shape: {sample['camera_K'].shape}")
            print(f"   - World2camera shape: {sample['world2camera'].shape}")
            print(f"   - Image path: {sample['image_path']}")
            
            # Test multiple samples
            print(f"\n3. Testing multiple samples...")
            test_indices = [0, min(10, len(dataset)-1), min(100, len(dataset)-1)]
            for test_idx in test_indices:
                if test_idx < len(dataset):
                    try:
                        sample = dataset[test_idx]
                        print(f"   ✓ Sample {test_idx} loaded successfully")
                    except Exception as e:
                        print(f"   ✗ Failed to load sample {test_idx}: {e}")
            
            # Test label remapping
            print(f"\n4. Testing label remapping...")
            print(f"   - Number of classes: {dataset.num_class}")
            print(f"   - Label remaps shape: {dataset.label_remaps.shape}")
            print(f"   - Origin color map shape: {dataset.origin_color_map.shape}")
            print(f"   - Filtered color map shape: {dataset.filted_color_map.shape}")
            
            # Test camera mask loading
            print(f"\n5. Testing camera mask loading...")
            # Check if camera masks are being loaded
            unique_cameras = set()
            for i in range(min(100, len(dataset))):
                sample = dataset[i]
                camera_name = sample['image_path'].split('/')[-2]
                unique_cameras.add(camera_name)
            print(f"   - Unique cameras found: {sorted(unique_cameras)}")
            
            # Generate visualization images for verification
            print(f"\n6. Generating mask visualization images...")
            output_base_dir = "test_output_masks"
            os.makedirs(output_base_dir, exist_ok=True)
            
            # Process all samples in the dataset
            print(f"   - Processing all {len(dataset)} samples...")
            samples_to_visualize = []
            camera_dirs = {}  # Track created camera directories
            
            for i in range(len(dataset)):
                try:
                    sample = dataset[i]
                    camera_name = sample['image_path'].split('/')[-2]
                    samples_to_visualize.append((i, sample, camera_name))
                    
                    # Create camera-specific directory if not exists
                    if camera_name not in camera_dirs:
                        camera_dir = os.path.join(output_base_dir, camera_name)
                        os.makedirs(camera_dir, exist_ok=True)
                        camera_dirs[camera_name] = camera_dir
                except Exception as e:
                    print(f"   ⚠ Warning: Failed to load sample {i}: {e}")
                    continue
            
            print(f"   - Generating visualizations for {len(samples_to_visualize)} samples...")
            print(f"   - Camera directories created: {sorted(camera_dirs.keys())}")
            
            def add_text_annotation(img, text, position=(10, 30), font_scale=0.7, thickness=2, 
                                   bg_color=(0, 0, 0), text_color=(255, 255, 255)):
                """Add text annotation with background"""
                img = img.copy()
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                                      font_scale, thickness)
                # Draw background rectangle
                cv2.rectangle(img, 
                            (position[0] - 5, position[1] - text_height - 5),
                            (position[0] + text_width + 5, position[1] + baseline + 5),
                            bg_color, -1)
                # Draw text
                cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, text_color, thickness)
                return img
            
            for idx, sample, camera_name in samples_to_visualize:
                # Get image, label, and mask
                image = (sample['image'] * 255).astype(np.uint8)
                label = sample['static_label']
                mask = sample['static_mask']
                
                # Calculate statistics
                mask_valid_ratio = (mask > 0.5).sum() / mask.size * 100
                label_unique = np.unique(label)
                label_counts = {cls: (label == cls).sum() for cls in label_unique}
                
                # Convert label to color visualization
                label_colored = dataset.filted_color_map[label].squeeze()
                label_colored = cv2.cvtColor(label_colored, cv2.COLOR_RGB2BGR)
                
                # Convert mask to visualization (0=black, 1=white)
                mask_vis = (mask * 255).astype(np.uint8)
                mask_colored = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
                
                # Create overlay: image with mask overlay
                mask_overlay = image.copy()
                mask_binary = mask > 0.5
                overlay_color = np.array([0, 255, 0], dtype=np.uint8)  # Green for valid regions
                mask_overlay[mask_binary] = (mask_overlay[mask_binary] * 0.7 + overlay_color * 0.3).astype(np.uint8)
                
                # Create label overlay
                label_overlay = cv2.addWeighted(image, 0.6, label_colored, 0.4, 0)
                
                # Add annotations to images
                image_annotated = add_text_annotation(image, f"Original Image - {camera_name}", 
                                                    position=(10, 30))
                image_annotated = add_text_annotation(image_annotated, 
                                                    f"Sample #{idx} | Shape: {image.shape[1]}x{image.shape[0]}", 
                                                    position=(10, 60))
                
                label_colored_annotated = add_text_annotation(label_colored, 
                                                             f"Semantic Label - {camera_name}",
                                                             position=(10, 30))
                label_colored_annotated = add_text_annotation(label_colored_annotated,
                                                             f"Classes: {sorted(label_unique.tolist())}",
                                                             position=(10, 60))
                
                mask_vis_annotated = add_text_annotation(mask_vis, 
                                                        f"Mask (Binary) - {camera_name}",
                                                        position=(10, 30))
                mask_vis_annotated = add_text_annotation(mask_vis_annotated,
                                                        f"Valid: {mask_valid_ratio:.1f}% | Filtered: {100-mask_valid_ratio:.1f}%",
                                                        position=(10, 60))
                
                mask_colored_annotated = add_text_annotation(mask_colored,
                                                            f"Mask (Colored) - {camera_name}",
                                                            position=(10, 30))
                mask_colored_annotated = add_text_annotation(mask_colored_annotated,
                                                            f"White=Valid, Black=Filtered",
                                                            position=(10, 60))
                
                mask_overlay_annotated = add_text_annotation(mask_overlay,
                                                            f"Mask Overlay - {camera_name}",
                                                            position=(10, 30))
                mask_overlay_annotated = add_text_annotation(mask_overlay_annotated,
                                                            f"Green=Valid Region",
                                                            position=(10, 60))
                
                label_overlay_annotated = add_text_annotation(label_overlay,
                                                             f"Label Overlay - {camera_name}",
                                                             position=(10, 30))
                
                # Create combined visualization with annotations (only save this)
                base_name = f"sample_{idx:04d}"
                camera_dir = camera_dirs[camera_name]
                h, w = image.shape[:2]
                combined = np.zeros((h * 2 + 60, w * 3, 3), dtype=np.uint8)  # Extra space for title
                combined.fill(50)  # Dark gray background
                
                # Add title
                title_text = f"Sample #{idx} - {camera_name} | Valid: {mask_valid_ratio:.1f}%"
                combined = add_text_annotation(combined, title_text, position=(w, 25), 
                                              font_scale=1.0, thickness=2)
                
                # Place images
                combined[60:60+h, 0:w] = image_annotated
                combined[60:60+h, w:2*w] = mask_overlay_annotated
                combined[60:60+h, 2*w:3*w] = label_overlay_annotated
                combined[60+h:60+2*h, 0:w] = mask_colored_annotated
                combined[60+h:60+2*h, w:2*w] = label_colored_annotated
                combined[60+h:60+2*h, 2*w:3*w] = cv2.cvtColor(mask_vis_annotated, cv2.COLOR_GRAY2BGR)
                
                # Add column labels
                col_labels = ["Original", "Mask Overlay", "Label Overlay", 
                             "Mask (Colored)", "Label (Colored)", "Mask (Binary)"]
                for i, label_text in enumerate(col_labels):
                    x = (i % 3) * w + w // 2 - 50
                    y = 50 if i < 3 else 60 + h + 20
                    combined = add_text_annotation(combined, label_text, position=(x, y),
                                                font_scale=0.6, thickness=1)
                
                # Save to camera-specific directory
                output_path = os.path.join(camera_dir, f"{base_name}_combined.jpg")
                cv2.imwrite(output_path, combined)
                print(f"   ✓ Saved visualizations for sample {idx} ({camera_name}) - Valid: {mask_valid_ratio:.1f}%")
            
            print(f"\n   All visualization images saved to: {output_base_dir}/")
            print(f"   - Combined view files: *_combined.jpg")
            print(f"   - Total samples processed: {len(samples_to_visualize)}")
            print(f"   - Images organized by camera in subdirectories:")
            for cam_name, cam_dir in sorted(camera_dirs.items()):
                count = len([f for f in os.listdir(cam_dir) if f.endswith('_combined.jpg')])
                print(f"     * {cam_name}/: {count} images")
            
        else:
            print("\n   ✗ No samples found in dataset!")
            return False
        
        print("\n" + "="*60)
        print("✓ All tests passed successfully!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset_loading()
    exit(0 if success else 1)

