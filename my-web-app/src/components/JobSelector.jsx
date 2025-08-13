import React, { useState, useEffect } from "react";
import "./JobSelector.jsx"
import HelpIcon from "./HelpIcon.jsx";

const BASE_API = "http://www.ihpc.se.ritsumei.ac.jp/detect";

function JobSelector({ selectedUrl, setSelectedUrl, selectedModel, setSelectedModel, selectedExp, setSelectedExp }) {
  const [options, setOptions] = useState([
    { name: "YOLOv8", value: `${BASE_API}/yolov8`, model: "yolov8s.pt", exp: null }
  ]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    async function fetchDoneJobs() {
      setLoading(true);
      try {
        const res = await fetch(`${BASE_API}/job_list`);
        const jobs = await res.json();

        // 并发请求所有job的训练状态
        const statusPromises = jobs.map(job =>
          fetch(`${BASE_API}/train_status/${job}`)
            .then(r => r.ok ? r.json() : Promise.resolve({}))
            .catch(() => ({}))
        );
        const statuses = await Promise.all(statusPromises);

        // 筛选状态为 done 的 job
        const doneJobs = jobs.filter((job, idx) => statuses[idx].status === "done");

        // 请求每个 done job 的最新训练结果 exp
        const expPromises = doneJobs.map(async (job) => {
          try {
            const res = await fetch(`${BASE_API}/datasets/${job}`);
            if (!res.ok) return { job, exp: null };
            const data = await res.json();
            const exps = data.experiments || [];
            if (exps.length === 0) return { job, exp: null };
            // 取最新 exp（字符串倒序）
            const latestExp = exps.sort().reverse()[0];
            return { job, exp: latestExp };
          } catch {
            return { job, exp: null };
          }
        });
        const jobsWithExp = await Promise.all(expPromises);

        const jobOptions = jobsWithExp.map(({ job, exp }) => ({
          name: job,
          value: `${BASE_API}/detect/${job}`,
          model: job,
          exp,
        }));

        setOptions([
          { name: "YOLOv8", value: `${BASE_API}/yolov8`, model: "yolov8s.pt", exp: null },
          ...jobOptions
        ]);

        // 只在首次加载时设置默认选项
        if (!selectedUrl) {
          setSelectedUrl(`${BASE_API}/yolov8`);
          setSelectedModel("yolov8s.pt");
          setSelectedExp(null);
        }
      } catch (e) {
        console.error("获取作业或状态失败", e);
      } finally {
        setLoading(false);
      }
    }
    fetchDoneJobs();
  }, []); // 空依赖，确保只执行一次

  return (
    <div className="model-select-inline">
      <label>モデル選択：
        <HelpIcon text="利用するモデルを選択してください。YOLOはリアルタイム物体検出のための高速なモデルで、job_xは対応するjobのデータセットで学習したカスタムモデルです。"/>
      </label>
      <select
        value={selectedUrl || ""}
        onChange={(e) => {
          const selected = options.find(opt => opt.value === e.target.value);
          if (selected) {
            setSelectedUrl(selected.value);
            setSelectedModel(selected.model);
            setSelectedExp(selected.exp);  // 这里设置 exp
          }
        }}
        disabled={loading}
      >
        {loading && <option>読み込み中...</option>}
        {!loading && options.length === 0 && <option>利用可能なモデルがありません</option>}
        {!loading && options.map(opt => (
          <option key={opt.value} value={opt.value}>
            {opt.name}
          </option>
        ))}
      </select>
    </div>
  );
}

export default JobSelector;
