import { Routes, Route } from "react-router-dom";
import Home from "../pages/home/home.jsx";

export default function AppRouter() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
    </Routes>
  );
}
