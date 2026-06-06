# isro

## Image Receiving + Matching

This project matches incoming test images against `ref1`, `ref2`, and `ref3`.

Incoming images are saved into:

```text
test/
```

Downscaled grayscale preview images are also saved separately for showing:

```text
down_scale_ref_images/
down_scale_test_images/
```

Detected images are copied into the folder for the best matching reference:

```text
ref1_detections/
ref2_detections/
ref3_detections/
```

Start the image receiver:

```bash
python3 image_receiver.py
```

Health check:

```text
http://<mac-ip>:8000/health
```

Upload an image:

```text
POST http://<mac-ip>:8000/upload?filename=test.png
```

The request body should be the raw image file bytes.

In another terminal, start matching images from the `test` folder:

```bash
python3 match.py --watch-test-folder
```

To process the current `test` folder once and exit:

```bash
python3 match.py --once
```
