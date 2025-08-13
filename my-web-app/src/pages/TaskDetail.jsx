import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import SampleImages from "../components/SampleImages";
import Header from "../components/Header";
import AugmentationPanel from "../components/AugmentationPanel";
import TrainingParameters from "../components/TrainingParameters";
import "./TaskDetail.css";
import AugmentationChart from "../components/AugmentationChart";
import ExperimentTabs from '../components/ExperimentTabs';
import HelpIcon from "../components/HelpIcon";

const BASE_API = "http://www.ihpc.se.ritsumei.ac.jp/detect";



const TaskDetail = () => {
  const { id } = useParams();
  const [info, setInfo] = useState(null);
  const [experiments, setExperiments] = useState([]); 
  const [selectedImage, setSelectedImage] = useState(null);
  const [augmentationSettings, setAugmentationSettings] = useState({});
  const [trainStatus, setTrainStatus] = useState("読み込み中...");
  const [augCounts, setAugCounts] = useState(null); 
  const [ActiveIndex, setActiveIndex] = useState(1);
  const [isTraining, setIsTraining] = useState(false);

  const [trainingInProgress, setTrainingInProgress] = useState(false);
  const [currentTrainingExperimentPath, setCurrentTrainingExperimentPath] = useState('');
  const [selectedEpoch, setSelectedEpoch] = useState(10); // 默认10
  const [optimizer, setOptimizer] = useState('AdamW'); // 默认Adam
  const [learningRate, setLearningRate] = useState(0.001); // 默认0.001
  const [batch, setBatch] = useState(16);
  const [weightDecay, setWeightDecay] = useState(0.0005);
  const [modelType, setModelType] = useState('yolov8n'); // 默认yolov8n
  const [usePretrained, setUsePretrained] = useState(true); // 默认使用预训练

  const handleStartTraining = async () => {
    const startTime = new Date().toISOString(); // 获取当前时间字符串
    setActiveIndex(0);
    try {
      const res = await fetch(`${BASE_API}/train`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          job_id: `job_${id}`,
          augmentations: augmentationSettings,
          start_time: startTime,
          epochs: selectedEpoch,
          optimizer: optimizer,
          learning_rate: learningRate,
          batch_size: batch,
          weight_decay: weightDecay,
          model_type: modelType,
          use_pretrained: usePretrained,
        }),

      });

      const result = await res.json();

      // 设置增强统计
      setAugCounts(result.augmentation_counts);

      // 设置“进行中”状态和路径
      setTrainingInProgress(true);
      const dateObj = new Date(startTime);
      const year = dateObj.getFullYear();
      const month = String(dateObj.getMonth() + 1).padStart(2, '0');
      const day = String(dateObj.getDate()).padStart(2, '0');
      const hours = String(dateObj.getHours()).padStart(2, '0');
      const minutes = String(dateObj.getMinutes()).padStart(2, '0');
      const seconds = String(dateObj.getSeconds()).padStart(2, '0');

      const currentExperimentPath = `job_${id}/results/exp_${year}${month}${day}_${hours}${minutes}${seconds}/results.csv`;

      setCurrentTrainingExperimentPath(currentExperimentPath);

      setTrainStatus("実行中");
    } catch (error) {
      console.error("トレーニング失敗:", error);
      alert("トレーニングの開始に失敗しました");
    }
  };

  const handleCancelTraining = async () => {
    const confirmed = window.confirm("キャンセルしますか？");
    if (!confirmed) return;

    try {
      const res = await fetch(`${BASE_API}/cancel_train/job_${id}`, {
        method: "POST",
      });
      const result = await res.json();
      alert(result.message || "キャンセル完了");
      setTrainStatus("キャンセル済み");
      setIsTraining(false);
    } catch (e) {
      alert("キャンセルに失敗しました");
    }
  };
useEffect(() => {
  const fetchStatus = () => {
    fetch(`${BASE_API}/train_status/job_${id}`)
      .then((res) => res.json())
      .then((data) => {
        const status = data.status;
        if (status === "running") {
          setTrainStatus("実行中");
          setIsTraining(true);
        } else if (status === "done") {
          setTrainStatus("完了");
          setIsTraining(false);
        } else if (status === "idle" || status === "cancelled") {
          setTrainStatus("中断");
          setIsTraining(false);
        } else {
          setTrainStatus("不明");
          setIsTraining(false);
        }

        if (data.augmentation_counts) {
          setAugCounts(data.augmentation_counts);
        }
      })
      .catch((err) => console.error("Status polling error:", err));
  };

  fetchStatus(); // 初次请求
  const intervalId = setInterval(fetchStatus, 5000); // 每5秒请求一次

  return () => clearInterval(intervalId); // 组件卸载时清除定时器
}, [id]);



  useEffect(() => {
    fetch(`${BASE_API}/datasets/job_${id}`)
      .then((res) => res.json())
      .then((data) => {
      setInfo(data); 
      if (data.experiments) {
          setExperiments(data.experiments);
        }
      if (data.sample_images && data.sample_images.length > 0) {
        setSelectedImage(`data:image/jpeg;base64,${data.sample_images[0].base64}`); // ✅ 默认选择第一张图
      }
    })
      .catch((err) => console.error("Status Error 1:", err));
  }, [id]);
  if (!info) return <p>読み込み中...</p>;

  return (
    <div className="task-detail-page">
      <Header/>
      <div className="task-detail-container">
        {/* 左侧控制区域 */}
        <div className="left-panel">
          <h2>{`job: ${id}`}</h2>
          <p>画像数: {info.image_count}  物体数: {info.label_count}</p>

          <h4>サンプル画像</h4>
          <SampleImages
            images={info.sample_images}
            onSelect={setSelectedImage}
          />
          <AugmentationPanel
            sampleImageBase64={selectedImage}
            augmentationSettings={augmentationSettings}
            setAugmentationSettings={setAugmentationSettings}
          />
          <TrainingParameters
            selectedEpoch={selectedEpoch}
            setSelectedEpoch={setSelectedEpoch}
            optimizer={optimizer}
            setOptimizer={setOptimizer}
            learningRate={learningRate}
            setLearningRate={setLearningRate}
            batch={batch}
            setBatch={setBatch}
            weightDecay={weightDecay}
            setWeightDecay={setWeightDecay}
            isTraining={isTraining}
            handleStartTraining={handleStartTraining}
            handleCancelTraining={handleCancelTraining}
            modelType={modelType}
            setModelType={setModelType}
            usePretrained={usePretrained}
            setUsePretrained={setUsePretrained}
          />
        </div>


        {/* 右侧状态区域 */}
        <div className="right-panel">
          <h3>トレーニング状態</h3>
          <p>現在のステータス: {trainStatus}</p>
          {(trainStatus === "実行中" || trainStatus === "完了") && augCounts && (
            <AugmentationChart data={augCounts} />
          )}

          <ExperimentTabs
            experiments={info.experiments}
            jobId={id}
            isTraining={isTraining}
            trainingExperimentPath={currentTrainingExperimentPath}
            trainingInProgress={trainingInProgress}
            activeIndex={ActiveIndex}
            setActiveIndex={setActiveIndex}
          />
        </div>
      </div>
    </div>
  );
};

export default TaskDetail;
