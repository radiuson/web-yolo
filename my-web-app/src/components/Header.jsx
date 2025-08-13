import React from "react";
import "./Header.css";

import ritsumeiImg from "../assets/images/ritsumei.jpg";
import logoImg from "../assets/images/ritsumei-logo.png";
import labImg from "../assets/images/lab-logo.png";

const HeaderImage = () => {
  return (
    <div className="header-container">
      {/* 背景图 */}
      <img
        src={ritsumeiImg}
        alt="Ritsumeikan Background"
        className="header-background"
      />

      {/* 左上 Logo */}
      <img
        src={logoImg}
        alt="Left Logo"
        className="logo-left"
      />

      {/* 右上 Lab 图标 */}
      <img
        src={labImg}
        alt="Right Logo"
        className="logo-right"
      />
    </div>
  );
};

export default HeaderImage;
