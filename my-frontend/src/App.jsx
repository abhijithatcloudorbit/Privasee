import { useState } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import SplashScreen from "./components/SplashScreen.jsx";
import Home from "./pages/home/Home.jsx";
import Login from "./pages/Auth/login.jsx";
import Signup from "./pages/Auth/signup.jsx";
import DashboardPage from "./pages/dashboard/DashboardPage.jsx"; // ← NEW

export default function App() {
  const [showSplash, setShowSplash] = useState(true);

  return (
    <>
      {/* Show splash only once */}
      {showSplash ? (
        <SplashScreen onFinish={() => setShowSplash(false)} />
      ) : (
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />

            {/* NEW — Guest Dashboard Route */}
            <Route path="/dashboard" element={<DashboardPage />} />
          </Routes>
        </BrowserRouter>
      )}
    </>
  );
}
