#!/bin/bash

# Package DeepSearchAI.app for distribution
APP_NAME="DeepSearchAI"
DIST_DIR="dist"
ZIP_NAME="${APP_NAME}_macOS.zip"

echo "📦 Packaging ${APP_NAME}.app..."

if [ -d "${DIST_DIR}/${APP_NAME}.app" ]; then
    # Remove old zip if it exists
    rm -f "${ZIP_NAME}"
    
    # Create new zip
    zip -r "${ZIP_NAME}" "${DIST_DIR}/${APP_NAME}.app"
    
    echo "✅ Success! Created ${ZIP_NAME}"
    echo "📍 You can now upload this file to GitHub Releases."
else
    echo "❌ Error: ${DIST_DIR}/${APP_NAME}.app not found."
    echo "Please run 'pyinstaller DeepSearchAI.spec' first."
fi
