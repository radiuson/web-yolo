import React, { useState, useEffect } from "react";
import Header from "../components/Header";
import { Link } from "react-router-dom";
import "./Train.css";

const BASE_API = "http://www.ihpc.se.ritsumei.ac.jp/detect";

const getStatusColor = (status) => {
  switch (status) {
    case "running":
      return "orange";
    case "done":
      return "green";
    case "failed":
      return "red";
    default:
      return "gray";
  }
};

const Train = () => {
  const itemsPerPage = 10;
  const [currentPage, setCurrentPage] = useState(1);
  const [jumpPage, setJumpPage] = useState("");
  const [taskList, setTaskList] = useState([]);

  useEffect(() => {
    const fetchTasks = async () => {
      try {
        const res = await fetch(`${BASE_API}/job_list`);
        const jobs = await res.json(); // 假设返回：["job_1", "job_2", ...]

        // 并发请求每个 job 的状态
        const statusPromises = jobs.map(async (jobName) => {
          const statusRes = await fetch(`${BASE_API}/train_status/${jobName}`);
          const statusData = await statusRes.json();
          return {
            name: jobName,
            id: jobName.replace("job_", ""),
            epoch: 0, // 如果你还没加 epoch 可以先写死或从别处获取
            status: statusData.status || "不明",
          };
        });

        const tasksWithStatus = await Promise.all(statusPromises);
        setTaskList(tasksWithStatus);
      } catch (err) {
        console.error("List error 0", err);
      }
    };

    fetchTasks();
  }, []);


  const totalPages = Math.ceil(taskList.length / itemsPerPage);
  const currentTasks = taskList.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const handleJump = () => {
    const page = parseInt(jumpPage);
    if (!isNaN(page) && page >= 1 && page <= totalPages) {
      setCurrentPage(page);
    }
    setJumpPage("");
  };

  return (
    <div className="train-page">
      <Header />

      {/* 分页控制栏 */}
      <div className="train-nav-bar">
        <button
          className="nav-btn"
          onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
          disabled={currentPage === 1}
        >
          ← 前へ
        </button>
        <span className="page-indicator">
          {currentPage} / {totalPages}
        </span>
        <button
          className="nav-btn"
          onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
          disabled={currentPage === totalPages}
        >
          次へ →
        </button>

        <div className="jump-container">
          <label>ページ移動:</label>
          <input
            type="number"
            min="1"
            max={totalPages}
            className="page-input"
            value={jumpPage}
            onChange={(e) => setJumpPage(e.target.value)}
          />
          <button className="jump-btn" onClick={handleJump}>
            移動
          </button>
        </div>
      </div>

      {/* 任务卡片区域 */}
      <div className="train-card-container">
        {currentTasks.map((task) => (
          <Link
            to={`/train/${task.id}`}
            key={task.id}
            style={{ textDecoration: "none", color: "inherit" }}
          >
            <div className="task-card">
              <h3>{task.name}</h3>
              <p>
                状態:{" "}
                <span style={{ color: getStatusColor(task.status) }}>
                  {task.status === "running"
                    ? "実行中"
                    : task.status === "done"
                    ? "完了"
                    : "待機"}
                </span>
              </p>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
};

export default Train;
