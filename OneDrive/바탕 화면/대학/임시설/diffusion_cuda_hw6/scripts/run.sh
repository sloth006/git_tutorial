echo "[Step 1] Sampling"
bin/sample
echo "[Step 1] Done!"

echo "[Step 2] Converting the random-seeded image to PNG format..."
cd utils
/nfs/home/proj1_env/bin/python3 img_convert.py
echo "[Step 2] Done!"

mv initial_img_0.png ../results/
mv sampled_img_0.png ../results/
echo "Please check the initial/generated image (initial/sampled_img_0.png) in the results directory."