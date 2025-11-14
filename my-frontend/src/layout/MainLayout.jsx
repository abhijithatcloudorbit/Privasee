import { useState } from "react";
import SplashScreen from "../components/SplashScreen";
import Navbar from "../components/layouts/Navbar.jsx";

export default function MainLayout({ children }) {
  return (
    <>
      <Navbar />

      <div style={{ paddingTop: "140px" }}>
        {children}
      </div>
    </>
  );
}
