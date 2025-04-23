# Manual Download Instructions for Model Files

If the automatic download script (`download_models.py`) doesn't work for you, follow these instructions to download the model files manually:

## Step 1: Download the Model Files

Go to the Google Drive link:
[https://drive.google.com/drive/folders/1azl9usQhT4mgJ-baMhwGuhdsb1_7DINc?usp=sharing](https://drive.google.com/drive/folders/1azl9usQhT4mgJ-baMhwGuhdsb1_7DINc?usp=sharing)

You should see three folders:
- densepose
- humanparsing
- openpose

Download each folder by right-clicking and selecting "Download".

## Step 2: Extract and Place Files

After downloading, extract the zip files and place the contents in the following structure:

```
ckpt/
├── densepose/
│   └── (all densepose model files)
├── humanparsing/
│   └── (all humanparsing model files)
└── openpose/
    └── ckpts/
        └── (all openpose model files)
```

## Step 3: Verify Directory Structure

Make sure the files are placed in the correct directories as shown above. The openpose models should go into the `ckpt/openpose/ckpts/` directory (note the additional `ckpts` subdirectory).

## Common Issues

If you encounter permission or access issues with Google Drive:
- Make sure you're logged into your Google account
- Try accessing the link in a different browser
- If the link appears broken, please contact the repository maintainer

When in doubt about the file structure, refer to the repository documentation or check the code that uses these model files to confirm the expected locations. 