import React, { useState } from "react";
import "./AdvancedOption.css"
const AdvancedOption = ({ title, children }) => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleOpen = () => setIsOpen(!isOpen);

  return (
    <div className="accordion">
      <button className="accordion-header" onClick={toggleOpen} aria-expanded={isOpen}>
        {title}
        <span className="accordion-icon">{isOpen ? "▲" : "▼"}</span>
      </button>
      {isOpen && <div className="accordion-content">{children}</div>}
    </div>
  );
};

export default AdvancedOption;
