# MyWebApp

一个基于 **Vite + React** 的前端项目，用于快速开发和部署现代 Web 应用。

## 📦 安装

1. **克隆仓库**
   ```bash
   git clone https://github.com/radiuson/web-yolo.git
   cd web-yolo

2. **安装依赖**
   使用 `npm` (推荐)：
   ```bash
   npm install
   ```

   或使用 `yarn`：
   ```bash
   yarn install
   ```

   或使用 `pnpm`：
   ```bash
   pnpm install
   ```

## 🚀 开发模式运行

```bash
npm run dev
```

运行后 Vite 会在本地启动开发服务器（默认端口为 `5173`），在浏览器中访问：
```
http://localhost:5173
```

支持热更新。

## 📦 构建生产版

```bash
npm run build
```

构建完成后，打包文件会输出到 `dist/` 目录，可以部署到静态网站。

## 🔍 预览生产版

```bash
npm run preview
```

## 📂 目录结构

```plaintext
mywebapp/
├── public/           # 公共静态资源
├── src/              # 源代码
│   ├── assets/       # 图片、字体等资源
│   ├── components/   # 公共组件
│   ├── pages/        # 页面组件
│   ├── App.jsx       # 应用入口组件
│   └── main.jsx      # 引导 React
├── index.html        # HTML 入口
├── package.json      # 项目信息与依赖
├── vite.config.js    # Vite 配置
└── README.md         # 项目说明
```

## ⚙ 环境要求

- **Node.js** >= 18
- **npm** / **yarn** / **pnpm**

可以使用 [nvm](https://github.com/nvm-sh/nvm) 管理 Node.js 版本

## 📄 许可证

MIT License
