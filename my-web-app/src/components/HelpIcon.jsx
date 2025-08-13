import React from "react";
import Tooltip from "@mui/material/Tooltip";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";

function HelpIcon({ text }) {
  return (
    <Tooltip title={text} arrow>
      <span
        style={{ display: "inline-flex", alignItems: "center", marginLeft: 6, cursor: "default" }}
        onClick={e => e.stopPropagation()}
      >
        <HelpOutlineIcon fontSize="small" />
      </span>
    </Tooltip>
  );
}

export default HelpIcon;
