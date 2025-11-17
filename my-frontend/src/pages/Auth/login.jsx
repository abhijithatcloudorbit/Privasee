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

    if (!validateEmail(email)) newErrors.email = true;
    if (!validatePassword(password)) newErrors.password = true;

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
          fontFamily: "Helvetica, Arial, sans-serif",
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

        {/* FORM FIELDS */}
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
          <FloatingInput
            label="Email"
            type="email"
            value={email}
            onChange={(val) => {
              setEmail(val);
              setErrors((prev) => ({ ...prev, email: !validateEmail(val) }));
            }}
            error={errors.email}
          />

          {/* PASSWORD FIELD */}
          <FloatingInput
            label="Password"
            type="password"
            value={password}
            onChange={(val) => {
              setPassword(val);
              setErrors((prev) => ({
                ...prev,
                password: !validatePassword(val),
              }));
            }}
            error={errors.password}
          />

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
            onClick={() => alert("Password recovery flow coming soon!")}
          >
            Forgot password?
          </p>

          {/* LOGIN BUTTON */}
          <ActionButton text="Login" onClick={handleLogin} />

          {/* OAUTH BUTTONS */}
          <div
            style={{
              marginTop: "40px",
              display: "flex",
              flexDirection: "row",
              gap: "20px",
              justifyContent: "center",
              flexWrap: "wrap",
            }}
          >
            <OAuthButton provider="google" label="Google" logo="/google.png" />
            <OAuthButton provider="github" label="GitHub" logo="/github.png" />
            <OAuthButton provider="outlook" label="Outlook" logo="/outlook.png" />
            <OAuthButton provider="apple" label="Apple" logo="/apple.png" />
          </div>
        </motion.div>
      </motion.div>
    </MainLayout>
  );
}

/* ========================= */
/*    FLOATING INPUT FIELD   */
/* ========================= */

function FloatingInput({ label, value, onChange, error, type }) {
  return (
    <div style={{ position: "relative", width: "350px" }}>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: "100%",
          padding: "18px 16px 10px 16px",
          borderRadius: "10px",
          border: error ? "2px solid red" : "2px solid #0d9488",
          outline: "none",
          fontSize: "1.1rem",
          color: error ? "red" : "#0d9488",
          fontFamily: "Helvetica",
          transition: "0.2s ease",
        }}
      />

      <label
        style={{
          position: "absolute",
          left: "16px",
          top: value ? "4px" : "16px",
          fontSize: value ? "0.75rem" : "1.1rem",
          color: error ? "red" : "#0d9488",
          transition: "all 0.2s ease",
        }}
      >
        {label}
      </label>
    </div>
  );
}

/* ========================= */
/*   MAIN LOGIN BUTTON       */
/* ========================= */

function ActionButton({ text, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        width: "200px",
        padding: "16px 0",
        backgroundColor: "#0d9488",
        color: "white",
        border: "2px solid transparent",
        borderRadius: "50px",
        fontSize: "1.3rem",
        fontWeight: "600",
        fontFamily: "Helvetica",
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
      {text}
    </button>
  );
}

/* ========================= */
/*     OAUTH BUTTONS         */
/* ========================= */

function OAuthButton({ provider, label, logo }) {
  return (
    <button
      onClick={() =>
        (window.location.href = `http://localhost:4000/auth/${provider}`)
      }
      style={{
        width: "260px",
        padding: "14px 20px",
        backgroundColor: "#0d9488",
        color: "white",
        border: "2px solid transparent",
        borderRadius: "50px",
        fontSize: "1.1rem",
        fontWeight: "600",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        cursor: "pointer",
        transition: "0.25s ease",
        fontFamily: "Helvetica",
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
      <span>Continue with {label}</span>
      <img src={logo} alt="" style={{ width: "22px" }} />
    </button>
  );
}
