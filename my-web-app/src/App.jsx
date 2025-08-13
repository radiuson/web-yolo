import React, { useState } from 'react';
import { HashRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { GrCamera, GrGraphQl } from 'react-icons/gr';
import { useNavigate } from 'react-router-dom';

import Home from './pages/Home';
import Train from './pages/Train';
import TaskDetail from "./pages/TaskDetail"; // ⬅️ 创建这个组件
function WelcomeScreen({ onStart }) {
  const navigate = useNavigate();
  const handleClick = () => {
    onStart(); // 通知 App 隐藏欢迎页
    navigate('/detect'); // 自动跳转
  };

  return (
    <div style={{
      height: '100vh',
      width: '100vw',
      background: 'linear-gradient(135deg, #f0f4f8, #e0e7ff)',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      fontFamily: 'Segoe UI, sans-serif',
      boxSizing: 'border-box',
      position: 'relative',  // 关键，作为定位上下文
      paddingBottom: '40px'  // 给底部留空间
    }}>
      <div style={{
        width: '90%',
        maxWidth: '960px',
        padding: '0 24px',
        textAlign: 'center'
      }}></div>
      <h1 style={{ fontSize: '2rem', color: '#1e40af' }}>Welcome to Ritsumeikan University YOLO Demo</h1>


      <button
        onClick={handleClick}
        style={{
          padding: '12px 24px',
          backgroundColor: '#1e40af',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          fontSize: '1rem',
          cursor: 'pointer'
        }}
      >
        Start Detection
      </button>

      <p style={{
        fontSize: '1rem',
        color: '#555',
        maxWidth: 500,
        textAlign: 'center',
        position: 'absolute',
        bottom: '10px',
        left: '50%',
        transform: 'translateX(-50%)',
        margin: 0
      }}>
        Powered by Intelligent High-performance Computing Laboratory.
      </p>
    </div>
  );
}
function BottomTab() {
  const location = useLocation();

  const tabs = [
    { path: '/detect', icon: <GrCamera size={24} style={{ color: 'steelblue' }} />, label: 'Detect' },
    { path: '/dashboard', icon: <GrGraphQl size={24} />, label: 'Train' },
  ];

  return (
    <div style={{
      position: 'fixed',
      bottom: 20,
      left: '50%',
      transform: 'translateX(-50%)',
      display: 'flex',
      gap: '40px',
      padding: '12px 24px',
      background: '#fff',
      borderRadius: 30,
      boxShadow: '0 4px 16px rgba(0,0,0,0.1)',
      zIndex: 1000
    }}>
      {tabs.map((tab) => {
        const active = location.pathname === tab.path;
        return (
          <Link
            key={tab.path}
            to={tab.path}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              color: active ? '#2563eb' : '#64748b',
              fontSize: 12,
              fontWeight: active ? 'bold' : 'normal',
              textDecoration: 'none'
            }}
          >
            {tab.icon}
            <span>{tab.label}</span>
          </Link>
        );
      })}
    </div>
  );
}

function App() {
  const [showWelcome, setShowWelcome] = useState(true);

  const handleStart = () => {
    setShowWelcome(false);
  };

  return (
    <Router>
      {showWelcome ? (
        <WelcomeScreen onStart={handleStart} />
      ) : (
        <>
          <div style={{ paddingBottom: '100px' }}>
            <Routes>
              <Route path="/detect" element={<Home />} />
              <Route path="/dashboard" element={<Train />} />
              <Route path="/train/:id" element={<TaskDetail />} />
            </Routes>
          </div>

          <BottomTab />
        </>
      )}
    </Router>
  );
}

export default App;
