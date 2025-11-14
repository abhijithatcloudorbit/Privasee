import { useState } from "react";
import SplashScreen from "../components/SplashScreen";

export default function MainLayout({ children }) {
  const [animationDone, setAnimationDone] = useState(false);

  return (
    <>
      {!animationDone && <IntroAnimation onFinish={() => setAnimationDone(true)} />}

      {animationDone && (
        <div>
          {/* TOP NAV */}
          <nav
            style={{
              height: "70px",
              display: "flex",
              alignItems: "center",
              padding: "0 20px",
              borderBottom: "1px solid #eee",
              position: "fixed",
              top: 0,
              left: 0,
              right: 0,
              background: "white",
              zIndex: 10,
            }}
          >
            <div style={{ fontSize: "24px", fontWeight: "700" }}>Privasee</div>
          </nav>

          <div style={{ paddingTop: "90px" }}>{children}</div>
        </div>
      )}
    </>
  );
}
