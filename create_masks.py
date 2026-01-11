import os
import cv2
import torch
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import gc

# --- CONFIGURATION ---
input_folder = "images"
output_folder = "masks"
confidence_threshold = 0.3
batch_size = 1        # Good balance for 5080 Laptop VRAM
chunk_size = 1       # Smaller chunks = more frequent RAM clearing
max_io_workers = 8     # Increased slightly for your 64GB RAM to speed up disk writes
# ---------------------

def save_mask(args):
    """Writes mask to disk."""
    path, mask_data = args
    cv2.imwrite(path, mask_data)

def main():
    # 1. Hardware Check
    if not torch.cuda.is_available():
        print("❌ Error: CUDA not detected.")
        return

    device = torch.device("cuda:0")
    print(f"🚀 Initializing {torch.cuda.get_device_name(0)}...")
    
    # 2. Load Model (UPDATED to YOLO11)
    # The first run will automatically download the weights (~60MB for 'x' model)
    try:
        model = YOLO('yolo11x-seg.pt').to(device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Hint: Run 'pip install -U ultralytics' to ensure YOLO11 support.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 3. File Discovery
    all_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    total_images = len(all_files)
    if total_images == 0:
        print("No images found.")
        return

    print(f"✅ Found {total_images} images. Starting processing with YOLO11...")

    # 4. Main Processing Loop with Progress Bar
    pbar = tqdm(total=total_images, desc="Total Progress", unit="img", mininterval=1.0)

    with ThreadPoolExecutor(max_workers=max_io_workers) as executor:
        for i in range(0, total_images, chunk_size):
            file_chunk = all_files[i : i + chunk_size]
            
            # Predict on the current chunk
            results = model.predict(
                source=file_chunk,
                conf=confidence_threshold,
                classes=[0],      # Detect people/class 0
                stream=True,      # Crucial: generators don't load everything to RAM
                device=device,
                batch=batch_size,
                retina_masks=True,
                verbose=False
            )

            for result in results:
                img_name = os.path.basename(result.path)
                h, w = result.orig_shape
                
                # Update progress bar description
                pbar.set_postfix(file=img_name[:15], refresh=True)
                
                # Initialize mask on GPU (255 = white background)
                final_mask_gpu = torch.full((h, w), 255, dtype=torch.uint8, device=device)

                if result.masks is not None:
                    # Combine masks on GPU
                    combined_mask = torch.any(result.masks.data > 0.5, dim=0)
                    
                    # Handle shape mismatch if retina_masks didn't return perfect size
                    if combined_mask.shape != (h, w):
                        combined_mask = torch.nn.functional.interpolate(
                            combined_mask.unsqueeze(0).unsqueeze(0).float(),
                            size=(h, w), mode='nearest'
                        ).squeeze()
                    
                    # Apply mask (0 = black object)
                    final_mask_gpu[combined_mask > 0.5] = 0
                
                # Transfer to CPU only for saving
                final_mask_cpu = final_mask_gpu.cpu().numpy()
                output_path = os.path.join(output_folder, f"{img_name}.mask.png")
                
                # Submit to background thread for disk writing
                executor.submit(save_mask, (output_path, final_mask_cpu))
                
                pbar.update(1)

            # --- HARD MEMORY CLEANUP ---
            del results
            torch.cuda.empty_cache()
            gc.collect() 

    pbar.close()
    print(f"\n✅ Done! {total_images} masks saved to: {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    main()
