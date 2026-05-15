# 3D Gaussian Splatting Model Viewer

This project is based on [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) for visualizing 3D Gaussian Splatting online. We provide a more user-friendly webpage for users to manage their 3D Gaussian Splats models (upload, delete)

For the basic operations of the 3DGS viewer, please refer to the [README in GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D/blob/main/README.md).

## Install node on Linux

```bash
# Following https://nodejs.org/en/download/package-manager
# installs nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source ~/.bashrc

# download and install Node.js (you may need to restart the terminal)
nvm install 22

# verifies the right Node.js version is in the environment
node -v # should print `v22.12.0`

# verifies the right npm version is in the environment
npm -v # should print `10.9.0`
```

### NOTE

There are a few tips to note before running the web demo:

- Configure the path at Line 19 in `server.cjs' to the project path of ModelZooWeb.

- You may also need to modify the camera axis at line 40 in `views/viewer.ejs` to better view your models.

- You may want to manually synchronize the viewer with the latest distribution of [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D). To do this, you can just simply copy the `src` folder and the `util` folder and replace the corresponding folders in this project.

## Building from source and running locally
Navigate to the code directory and run
```sh
npm install
# Or run: 
#  NODE_TLS_REJECT_UNAUTHORIZED=0 npm install
# in case there is a certification issue.

```
Next run the build. For Linux & Mac OS systems run:
```sh
npm run build
```
For Windows I have added a Windows-compatible version of the build command:
```sh
npm run build-windows
```
To view the demo scenes locally run
```sh
npm run demo
```
The demo will be accessible locally at [http://127.0.0.1:8080](http://127.0.0.1:8080). You will need to upload your models follow the instructions below: 
TODO(chenyu): model upload instructions.

The demo scene data is available here: [https://projects.markkellogg.org/downloads/gaussian_splat_data.zip](https://projects.markkellogg.org/downloads/gaussian_splat_data.zip)
<br>
<br>
