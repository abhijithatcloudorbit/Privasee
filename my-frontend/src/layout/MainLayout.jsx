import { useState } from "react";
import SplashScreen from "../components/SplashScreen";
import Navbar from "../components/layouts/Navbar.jsx";

export default function MainLayout({ children }) {
  const [animationDone, setAnimationDone] = useState(false);

  return (
    <>
      {/* Splash Screen */}
      {!animationDone && (
        <SplashScreen onFinish={() => setAnimationDone(true)} />
      )}

      {/* Main Application Layout */}
      {animationDone && (
        <>
          <Navbar />

          <div style={{ paddingTop: "140px" }}>
            {children}
          </div>
        </>
      )}
    </>
  );
}
