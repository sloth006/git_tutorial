

echo "[Step 1] Running basic unit test..."
bin/functionality
echo "[Step 1] Done!"

echo "[Step 2] Running performance test and generating a seed-fixed image..."
bin/sample -fs -s 7524
echo "[Step 2] Done!"

cd utils
echo "[Step 3] Converting the seed-fixed image to PNG format..."
/nfs/home/proj1_env/bin/python3 img_convert.py
echo "[Step 3] Done!"

echo "[Step 4] Compare the end-to-end results with the expected results..."
/nfs/home/proj1_env/bin/python3 e2e_check.py
echo "[Step 4] Done!"

mv initial_img_0.png ../results/
mv sampled_img_0.png ../results/
echo "Please check the generated image (results/sampled_img_0.png)."








