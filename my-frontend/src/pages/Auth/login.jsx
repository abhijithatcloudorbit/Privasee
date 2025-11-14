import MainLayout from "@/layout/MainLayout.jsx";
import { useState } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";

export default function Login() {
  const navigate = useNavigate();

  // Input states
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  // UI states
  const [shake, setShake] = useState(false);
  const [errors, setErrors] = useState({});
  const [success, setSuccess] = useState(false);

  // VALIDATION RULES
  const validateEmail = (email) =>
    /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

  const validatePassword = (password) =>
    password.length >= 6;

  const handleLogin = () => {
    let newErrors = {};

    if (!validateEmail(email)) {
      newErrors.email = "Invalid email format";
    }
    if (!validatePassword(password)) {
      newErrors.password = "Password must be at least 6 characters";
    }

    setErrors(newErrors);

    if (Object.keys(newErrors).length > 0) {
      setShake(true);
      setTimeout(() => setShake(false), 400);
      return;
    }

    // SUCCESS
    setSuccess(true);
    setTimeout(() => navigate("/dashboard"), 1800);
  };

  return (
    <MainLayout>
      {/* SUCCESS OVERLAY */}
      {success && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          style={{
            position: "fixed",
            inset: 0,
            backgroundColor: "#0d9488",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 999,
          }}
        >
          <h1
            style={{
              color: "white",
              fontFamily: "Helvetica",
              fontSize: "3rem",
              fontWeight: "700",
            }}
          >
            Login Successful 🎉
          </h1>
        </motion.div>
      )}

      {/* MAIN FORM */}
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
            fontSize: "4rem",
            fontWeight: "700",
            marginBottom: "30px",
          }}
        >
          Welcome back!
        </h1>

        <motion.div
          animate={shake ? { x: [-12, 12, -12, 12, 0] } : {}}
          transition={{ duration: 0.3 }}
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "32px",
          }}
        >
          {/* EMAIL FIELD */}
          <div style={{ position: "relative", width: "350px" }}>
            <input
              type="email"
              value={email}
              onChange={(e) => {
                setEmail(e.target.value);
                setErrors((prev) => ({
                  ...prev,
                  email: !validateEmail(e.target.value),
                }));
              }}
              style={{
                width: "100%",
                padding: "18px 16px 10px 16px",
                borderRadius: "10px",
                border: errors.email ? "2px solid red" : "2px solid #0d9488",
                outline: "none",
                fontSize: "1.1rem",
                color: errors.email ? "red" : "#0d9488",
                fontFamily: "Helvetica",
                transition: "0.2s ease",
              }}
            />

            {/* Floating Label */}
            <label
              style={{
                position: "absolute",
                left: "16px",
                top: email ? "4px" : "16px",
                fontSize: email ? "0.75rem" : "1.1rem",
                color: errors.email ? "red" : "#0d9488",
                transition: "all 0.2s ease",
              }}
            >
              Email
            </label>
          </div>

          {/* PASSWORD FIELD — NO EYE ICON */}
          <div style={{ position: "relative", width: "350px" }}>
            <input
              type="password" // Always masked
              value={password}
              onChange={(e) => {
                setPassword(e.target.value);
                setErrors((prev) => ({
                  ...prev,
                  password: !validatePassword(e.target.value),
                }));
              }}
              style={{
                width: "100%",
                padding: "18px 16px 10px 16px",
                borderRadius: "10px",
                border: errors.password ? "2px solid red" : "2px solid #0d9488",
                outline: "none",
                fontSize: "1.1rem",
                color: errors.password ? "red" : "#0d9488",
                fontFamily: "Helvetica",
                transition: "0.2s ease",
              }}
            />

            {/* Floating Label */}
            <label
              style={{
                position: "absolute",
                left: "16px",
                top: password ? "4px" : "16px",
                fontSize: password ? "0.75rem" : "1.1rem",
                color: errors.password ? "red" : "#0d9488",
                transition: "all 0.2s ease",
              }}
            >
              Password
            </label>
          </div>

          {/* FORGOT PASSWORD */}
          <p
            style={{
              fontSize: "1rem",
              color: "#0d9488",
              cursor: "pointer",
              marginTop: "-10px",
              marginBottom: "10px",
              fontWeight: "600",
            }}
            onClick={() =>
              alert("Password recovery flow will be implemented soon!")
            }
          >
            Forgot password?
          </p>

          {/* LOGIN BUTTON */}
          <button
            onClick={handleLogin}
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

          {/* OAUTH BUTTONS */}
          <div style={{ marginTop: "20px", display: "flex", gap: "20px" }}>
            <button
              style={{
                padding: "12px 20px",
                borderRadius: "50px",
                border: "2px solid #0d9488",
                background: "white",
                color: "#0d9488",
                fontWeight: "600",
                cursor: "pointer",
              }}
            >
              Continue with Google
            </button>

            <button
              style={{
                padding: "12px 20px",
                borderRadius: "50px",
                border: "2px solid #0d9488",
                background: "white",
                color: "#0d9488",
                fontWeight: "600",
                cursor: "pointer",
              }}
            >
              Continue with GitHub
            </button>
          </div>
        </motion.div>
      </motion.div>
    </MainLayout>
  );
}
