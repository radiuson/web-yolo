import React from 'react';

const ImageUploader = ({ image, setImage, setImageBase64 }) => {
  const fileInputRef = React.useRef(null);

  const handleImageChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();

    const maxWidth = 800; // 最大宽度
    const maxHeight = 800; // 最大高度
    const quality = 0.8; // 压缩质量（0~1）

    const img = new Image();
    img.src = URL.createObjectURL(file);

    img.onload = async () => {
      const canvas = document.createElement("canvas");

      let width = img.width;
      let height = img.height;

      if (width > maxWidth || height > maxHeight) {
        const scale = Math.min(maxWidth / width, maxHeight / height);
        width = width * scale;
        height = height * scale;
      }

      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, width, height);

      canvas.toBlob((blob) => {
        const compressedFile = new File([blob], file.name, { type: file.type });
        setImage(compressedFile); // ✅ 设置压缩后的 File 文件
        reader.onloadend = () => {
          if (setImageBase64) {
            setImageBase64(reader.result); // ✅ 设置 base64 字符串
          }
        };
        reader.readAsDataURL(blob); // ✅ 转 base64
      }, file.type, quality);
    };
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="mb-6">
      <button
        type="button"
        onClick={triggerFileInput}
        className="inline-flex items-center px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md shadow hover:bg-blue-700 transition duration-200"
      >
        画像を選択
      </button>

      <input
        type="file"
        accept="image/*"
        ref={fileInputRef}
        onChange={handleImageChange}
        style={{ display: 'none' }}
      />
    </div>
  );
};

export default ImageUploader;
