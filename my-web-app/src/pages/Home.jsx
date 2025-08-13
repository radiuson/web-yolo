import React, { useState } from "react";
import Header from "../components/Header";
import ImageUploader from "../components/ImageUploader";
import SliderBlock from "../components/SliderBlock";
import ImagePreview from "../components/ImagePreview";
import "./Home.css";
import JobSelector from "../components/JobSelector";
import HelpIcon from "../components/HelpIcon";


const Home = () => {
  const [image, setImage] = useState(null);
  const [confidence, setConfidence] = useState(50);
  const [iou, setIou] = useState(50);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [resultImageUrl, setResultImageUrl] = useState(null);
  const [selectedUrl, setSelectedUrl] = useState("http://www.ihpc.se.ritsumei.ac.jp/detect/yolov8");
  const [selectedModel, setSelectedModel] = useState("yolov8s.pt");
  const [imageBase64, setImageBase64] = useState(null);
  const [selectedExp, setSelectedExp] = useState(null);  // 新增

  const handleSubmit = async () => {
    if (!imageBase64) return;

    setLoading(true);
    setMessage("");
    setResultImageUrl(null);

    try {
      const body = {
        image_base64: imageBase64,
        confidence,
        iou,
      };
      // 只有当 selectedExp 有值时才传 exp 字段
      if (selectedExp) {
        body.exp = selectedExp;
      }
      console.log({body})
      const res = await fetch(selectedUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();

      setMessage(data.message || "検出成功");
      const base64Image = `data:image/jpeg;base64,${data.image_base64}`;
      setResultImageUrl(base64Image);
    } catch (err) {
      setMessage("検出に失敗しました");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="home-page">
      <Header />
      <div className="panel">
        <div className="uploader-model-row">
          <ImageUploader image={image} setImage={setImage} setImageBase64={setImageBase64} />
            <JobSelector
            selectedUrl={selectedUrl}
            setSelectedUrl={setSelectedUrl}
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
            selectedExp={selectedExp}
            setSelectedExp={setSelectedExp}
            />
        </div>

        <SliderBlock
          title="信頼値閾値"
          value={confidence}
          onChange={setConfidence}
          helpText="信頼値の閾値は、低い信頼度の検出結果をフィルタリングするための値です。数値が高いほど、より厳しくフィルタリングされます。"
        />

        <SliderBlock
          title="IoU閾値"
          value={iou}
          onChange={setIou}
          helpText="IoU（Intersection over Union）は、検出ボックスの重なり具合を示す指標です。閾値を高く設定すると、より厳密に重なりがある検出のみを有効とします。"
        />



        <div className="button-group">
          <button
            onClick={() => {
              setImage(null);
              setResultImageUrl(null);
              setMessage("");
            }}
            className={`btn-danger ${!image ? "disabled" : ""}`}
            disabled={!image}
          >
            画像削除
          </button>
          <button
            onClick={handleSubmit}
            className={`btn-submit ${!image ? "disabled" : ""}`}
            disabled={!image}
          >
            検出
          </button>
        </div>
      </div>

      <div className="image-area">
        {resultImageUrl ? (
          <ImagePreview imageUri={resultImageUrl} />
        ) : (
          image && <ImagePreview imageUri={URL.createObjectURL(image)} />
        )}

        <div className="status-area">
          {loading && <p style={{ color: "gray" }}>検出中...</p>}
          {!loading && message && <p style={{ color: "green" }}>{message}</p>}
        </div>
      </div>
    </div>
  );
};

export default Home;
