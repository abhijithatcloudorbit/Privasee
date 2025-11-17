import { useNavigate } from "react-router-dom";
import MainLayout from "@/layout/MainLayout.jsx";
import { motion } from "framer-motion";

export default function Home() {
  const navigate = useNavigate();

  return (
    <MainLayout>
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        style={{
          paddingTop: "40px",
          textAlign: "center",
          fontFamily: "Helvetica, Arial, Sans-Serif",
          color: "#0d9488",
        }}
      >
        <h1
          style={{
            fontSize: "6rem",
            fontWeight: "700",
            marginBottom: "10px",
          }}
        >
          Welcome to Privasee
        </h1>

        <p
          style={{
            fontSize: "2rem",
            marginTop: 0,
          }}
        >
          Let us help you in the boring part of your job!
        </p>

        {/* BUTTONS */}
        <div
          style={{
            marginTop: "40px",
            display: "flex",
            justifyContent: "center",
            gap: "40px",
          }}
        >
          {/* LOGIN BUTTON */}
          <button
            onClick={() => navigate("/login")}
            style={{
              width: "200px",
              padding: "16px 0",
              backgroundColor: "#0d9488",
              color: "white",
              border: "2px solid transparent",
              borderRadius: "50px",
              fontSize: "1.3rem",
              fontWeight: "600",
              fontFamily: "Helvetica, Arial, sans-serif",
              cursor: "pointer",
              transition: "0.25s ease",
            }}
            onMouseEnter={(e) => {
              e.target.style.backgroundColor = "white";
              e.target.style.color = "#0d9488";
              e.target.style.border = "2px solid #0d9488";
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = "#0d9488";
              e.target.style.color = "white";
              e.target.style.border = "2px solid transparent";
            }}
          >
            Login
          </button>

          {/* SIGN UP BUTTON */}
          <button
            onClick={() => navigate("/signup")}
            style={{
              width: "200px",
              padding: "16px 0",
              backgroundColor: "#0d9488",
              color: "white",
              border: "2px solid transparent",
              borderRadius: "50px",
              fontSize: "1.3rem",
              fontWeight: "600",
              fontFamily: "Helvetica, Arial, sans-serif",
              cursor: "pointer",
              transition: "0.25s ease",
            }}
            onMouseEnter={(e) => {
              e.target.style.backgroundColor = "white";
              e.target.style.color = "#0d9488";
              e.target.style.border = "2px solid #0d9488";
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = "#0d9488";
              e.target.style.color = "white";
              e.target.style.border = "2px solid transparent";
            }}
          >
            Sign Up
          </button>
                    {/* Continue as Guest */}
          <button
            onClick={() => navigate("/dashboard")}
            style={{
              background: "#0d9488",
              color: "#fff",
              padding: "14px 34px",
              borderRadius: 999,
              border: "none",
              fontWeight: 700,
              fontSize: "1.3rem",
              cursor: "pointer",
            }}
            onMouseEnter={(e) => {
              e.target.style.backgroundColor = "white";
              e.target.style.color = "#0d9488";
              e.target.style.border = "2px solid #0d9488";
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = "#0d9488";
              e.target.style.color = "white";
              e.target.style.border = "2px solid transparent";
            }}            
          >
            Continue as Guest
          </button>
        </div>
      </motion.div>
    </MainLayout>
  );
}
