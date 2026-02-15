# DeepSearch AI: Online Hosting & Distribution Guide

This guide explains how to share your project on GitHub and provide a downloadable version for users.

## 1. Prepare for GitHub

Ensure your project is clean and ready for public view.

### Check `.gitignore`
Your `.gitignore` is already set up to exclude large files (like `embeddings/` and `.cache/`) and sensitive info (`settings.json`). This is perfect.

### Update `README.md`
Make sure your `README.md` clearly describes the project and how to use it. (I will update this for you in the next step).

## 2. Push to GitHub

If you haven't already pushed your latest changes:

```bash
git add .
git commit -m "Prepare for public release"
git push origin main
```

## 3. Package the macOS App

To let others download and use your app without installing Python, you should provide the `.app` bundle from your `dist/` folder.

### Step-by-Step Packaging:
1.  Open your terminal in the project root.
2.  Run the following command to create a zip file:
    ```bash
    zip -r DeepSearchAI_macOS.zip dist/DeepSearchAI.app
    ```
    *(Alternatively, use the provided `package_app.sh` script).*

## 4. Create a GitHub Release

This is the professional way to distribute your app.

1.  Go to your repository on GitHub: `https://github.com/omiii18/ai-image-search`.
2.  On the right sidebar, click **"Releases"** -> **"Create a new release"**.
3.  **Tag version**: e.g., `v1.0.0`.
4.  **Release title**: `DeepSearch AI v1.0.0 Beta`.
5.  **Description**: Briefly list features or "Initial Release".
6.  **Attach Binaries**: Drag and drop the `DeepSearchAI_macOS.zip` you created in Step 3.
7.  Click **"Publish release"**.

---

### Important Note for Users
Since your app is not digitally signed with an Apple Developer account, users will need to:
1.  Download and unzip the app.
2.  **Right-click** on `DeepSearchAI.app` and select **"Open"**.
3.  Click **"Open"** again in the security prompt.
