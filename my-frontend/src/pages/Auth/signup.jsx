
import MainLayout from "@/layout/MainLayout.jsx";
import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";

/* =========================
   All-in-one Premium Signup
   ========================= */
export default function SignupAdvanced() {
  const navigate = useNavigate();

  // steps: 1=email, 2=password, 3=profile, 4=security/finish
  const [step, setStep] = useState(1);

  // core fields
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [firstName, setFirst] = useState("");
  const [lastName, setLast] = useState("");
  const [inviteCodeVisible, setInviteVisible] = useState(false);
  const [inviteCode, setInviteCode] = useState("");
  const [referral, setReferral] = useState("");
  const [newsletter, setNewsletter] = useState(true);
  const [agreeTerms, setAgreeTerms] = useState(false);
  const [showTermsModal, setShowTermsModal] = useState(false);
  const [termsScrolledToBottom, setTermsScrolledToBottom] = useState(false);
  const [securityQuestion, setSecurityQuestion] = useState("");
  const [securityAnswer, setSecurityAnswer] = useState("");
  const [guestMode, setGuestMode] = useState(false);

  // avatar
  const [avatarType, setAvatarType] = useState("initials"); // initials | emoji | upload
  const [emoji, setEmoji] = useState("😎");
  const [uploadedAvatar, setUploadedAvatar] = useState(null);

  // UI states
  const [errors, setErrors] = useState({});
  const [shake, setShake] = useState(false);
  const [success, setSuccess] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [progressCircle, setProgressCircle] = useState(true);

  // device/timezone info
  const [clientInfo, setClientInfo] = useState({ tz: "", locale: "", browser: "", os: "" });

  // password strength + rules
  const passwordRules = [
    { key: "length8", test: (p) => p.length >= 8, label: "At least 8 characters" },
    { key: "upper", test: (p) => /[A-Z]/.test(p), label: "Uppercase letter" },
    { key: "lower", test: (p) => /[a-z]/.test(p), label: "Lowercase letter" },
    { key: "number", test: (p) => /[0-9]/.test(p), label: "Number" },
    { key: "symbol", test: (p) => /[^A-Za-z0-9]/.test(p), label: "Symbol (!@#$...)" },
  ];

  // refs for autofocus
  const emailRef = useRef(null);
  const passwordRef = useRef(null);
  const confirmRef = useRef(null);
  const firstNameRef = useRef(null);

  // password strength score
  const getStrength = (p) => {
    let s = 0;
    if (p.length >= 8) s++;
    if (p.length >= 12) s++;
    if (/[A-Z]/.test(p)) s++;
    if (/[0-9]/.test(p)) s++;
    if (/[^A-Za-z0-9]/.test(p)) s++;
    return s; // 0-5
  };
  const strength = getStrength(password);

  // username suggestion from email (on blur or change)
  useEffect(() => {
    if (!username && email.includes("@")) {
      const local = email.split("@")[0].replace(/[^a-zA-Z0-9._]/g, "");
      const suggestion = local.length ? local : `user${Math.floor(Math.random()*900+100)}`;
      setUsername(suggestion);
    }
    // eslint-disable-next-line
  }, [email]);

  // client info detection on mount
  useEffect(() => {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || "";
    const locale = navigator.language || navigator.userLanguage || "en-US";
    const ua = navigator.userAgent || "";
    let browser = "Unknown";
    if (ua.includes("Chrome") && !ua.includes("Edge")) browser = "Chrome";
    else if (ua.includes("Safari") && !ua.includes("Chrome")) browser = "Safari";
    else if (ua.includes("Firefox")) browser = "Firefox";
    else if (ua.includes("Edg") || ua.includes("Edge")) browser = "Edge";
    let os = "Unknown";
    if (ua.includes("Win")) os = "Windows";
    else if (ua.includes("Mac")) os = "macOS";
    else if (ua.includes("Linux")) os = "Linux";
    else if (ua.includes("Android")) os = "Android";
    else if (/iPhone|iPad|iPod/.test(ua)) os = "iOS";
    setClientInfo({ tz, locale, browser, os });
  }, []);

  // helper validations
  const validateEmail = (e) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e);
  const validatePassword = (p) => p.length >= 8 && passwordRules.every(r => r.test(p));
  const validateConfirm = (c) => c === password;

  // mock username availability check (replace with backend)
  const checkUsernameAvailability = async (name) => {
    // MOCK: treat names containing "taken" as unavailable
    await new Promise((r) => setTimeout(r, 250));
    if (!name) return false;
    return !/taken/i.test(name);
  };

  // advance step with validation
  const next = async () => {
    let newErrors = {};
    if (step === 1) {
      if (!validateEmail(email)) newErrors.email = "Invalid email";
      // check username availability
      const ok = await checkUsernameAvailability(username);
      if (!ok) newErrors.username = "Username unavailable";
    }
    if (step === 2) {
      // check password rules
      passwordRules.forEach((r) => {
        if (!r.test(password)) newErrors[r.key] = r.label;
      });
      if (!validateConfirm(confirm)) newErrors.confirm = "Passwords do not match";
    }
    if (step === 3) {
      if (!firstName) newErrors.firstName = "Required";
      if (!agreeTerms) newErrors.terms = "Must agree to terms";
      if (!termsScrolledToBottom) newErrors.termsScroll = "Please read the terms";
    }

    setErrors(newErrors);

    if (Object.keys(newErrors).length > 0) {
      setShake(true);
      setTimeout(() => setShake(false), 400);
      return;
    }

    setStep((s) => Math.min(4, s + 1));

    // autofocus next field after motion settles
    setTimeout(() => {
      if (step === 1) passwordRef.current?.focus();
      if (step === 2) firstNameRef.current?.focus();
    }, 220);
  };

  // go back
  const back = () => {
    setStep((s) => Math.max(1, s - 1));
  };

  // final submit
  const submitSignup = () => {
    // Here you would call your API to create account -> use fetch/axios
    // We'll simulate success
    setSuccess(true);
    setTimeout(() => navigate("/login"), 2500);
  };

  // avatar upload preview
  const handleAvatarUpload = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => setUploadedAvatar(e.target.result);
    reader.readAsDataURL(file);
  };

  // simple countdown redirect display (after success)
  const [countdown, setCountdown] = useState(3);
  useEffect(() => {
    if (!success) return;
    const t = setInterval(() => setCountdown((c) => c - 1), 1000);
    return () => clearInterval(t);
  }, [success]);

  useEffect(() => {
    if (countdown <= 0 && success) {
      navigate("/login");
    }
  }, [countdown, success, navigate]);

  // small helper UI components inline to keep single-file
  const ProgressCircle = ({ progress }) => {
    const size = 80;
    const stroke = 6;
    const radius = (size - stroke) / 2;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (progress / 100) * circumference;
    return (
      <svg width={size} height={size} style={{ display: "block", margin: "0 auto" }}>
        <circle
          stroke="#e6e6e6"
          fill="transparent"
          strokeWidth={stroke}
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
        <circle
          stroke="#0d9488"
          fill="transparent"
          strokeWidth={stroke}
          strokeLinecap="round"
          r={radius}
          cx={size / 2}
          cy={size / 2}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 0.4s ease, stroke 0.4s ease" }}
        />
      </svg>
    );
  };

  // onboarding checklist (simple)
  const checklist = {
    email: validateEmail(email),
    username: username && username.length > 2,
    password: passwordRules.every((r) => r.test(password)),
    confirm: password === confirm && confirm.length > 0,
    profile: firstName.length > 0,
    terms: agreeTerms && termsScrolledToBottom,
  };

  // small confetti (simple CSS dot burst)
  const Confetti = () => (
    <div style={{ position: "absolute", inset: 0, pointerEvents: "none", zIndex: 1200 }}>
      <div style={{ position: "absolute", left: "30%", top: "20%", opacity: 0.9 }}>
        <span style={{ display: "inline-block", transform: "rotate(15deg)", fontSize: 22 }}>🎉</span>
      </div>
      <div style={{ position: "absolute", left: "50%", top: "10%", opacity: 0.8 }}>
        <span style={{ display: "inline-block", transform: "rotate(-10deg)", fontSize: 28 }}>✨</span>
      </div>
    </div>
  );

  // small helper to render OAuth buttons row (redirect to backend)
const OAuthButtonsRow = () => {
  const providers = [
    { id: "google", label: "Continue with Google", logo: "/google.png" },
    { id: "github", label: "Continue with GitHub", logo: "/github.png" },
    { id: "outlook", label: "Continue with Outlook", logo: "/outlook.png" },
    { id: "apple", label: "Continue with Apple", logo: "/apple.png" },
  ];

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        gap: 20,
        flexWrap: "nowrap",
        width: "100%",
        marginTop: 20,
      }}
    >
      {providers.map((p) => (
        <button
          key={p.id}
          onClick={() =>
            (window.location.href = `http://localhost:4000/auth/${p.id}`)
          }
          style={{
            width: 240,
            padding: "12px 20px",
            borderRadius: 40,
            border: "none",
            background: "#0d9488",
            color: "#fff",
            fontWeight: 600,
            fontFamily: "Helvetica",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 10,
            cursor: "pointer",
            transition: "0.25s ease",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "#fff";
            e.currentTarget.style.color = "#0d9488";
            e.currentTarget.style.border = "2px solid #0d9488";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "#0d9488";
            e.currentTarget.style.color = "#fff";
            e.currentTarget.style.border = "none";
          }}
        >
          <span>{p.label}</span>
          <img src={p.logo} alt={p.label} style={{ width: 18, height: 18 }} />
        </button>
      ))}
    </div>
  );
};


  // render
  return (
    <MainLayout>
      <div style={{ position: "relative" }}>
        {success && <Confetti />}

        {/* success overlay */}
        {success && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            style={{
              position: "fixed",
              inset: 0,
              backgroundColor: "#0d9488",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 1000,
            }}
          >
            <div style={{ textAlign: "center", color: "#fff" }}>
              <h1 style={{ fontFamily: "Helvetica", fontSize: 36, marginBottom: 8 }}>
                Welcome, {firstName || username} 🎉
              </h1>
              <p style={{ opacity: 0.95, fontSize: 18 }}>
                Finalizing your workspace... Redirecting in {countdown}
              </p>
            </div>
          </motion.div>
        )}

        {/* header + progress */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 24,
            justifyContent: "center",
            marginTop: 14
          }}
        >
          {progressCircle ? (
            <ProgressCircle progress={(step / 4) * 100} />
          ) : (
            <div style={{ width: "40%", minWidth: 220 }}>
              <div style={{ height: 6, background: "#e6e6e6", borderRadius: 6 }}>
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(step / 4) * 100}%` }}
                  transition={{ duration: 0.35 }}
                  style={{ height: "100%", background: "#0d9488", borderRadius: 6 }}
                />
              </div>
            </div>
          )}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45 }}
          style={{
            paddingTop: 24,
            textAlign: "center",
            fontFamily: "Helvetica",
            color: "#0d9488"
          }}
        >
          <h1 style={{ fontSize: 32, marginBottom: 8, fontWeight: 700 }}>
            Create your Privasee account
          </h1>
          <p style={{ color: "#0d9488", opacity: 0.85, marginBottom: 18 }}>
            Fast, secure and private — get started in seconds.
          </p>

          {/* Form container */}
          <motion.div
            animate={shake ? { x: [-8, 8, -8, 8, 0] } : {}}
            transition={{ duration: 0.35 }}
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 20
            }}
          >

            {/* ======================= */}
            {/*           STEP 1        */}
            {/* ======================= */}
            {step === 1 && (
              <>
                <FloatingInput
                  label="Email"
                  type="email"
                  value={email}
                  onChange={(v) => {
                    setEmail(v);
                    if (!username) {
                      const local = v.split("@")[0].replace(/[^a-zA-Z0-9._]/g, "");
                      setUsername(local || `user${Math.floor(Math.random() * 900 + 100)}`);
                    }
                    setErrors((p) => ({ ...p, email: !validateEmail(v) }));
                  }}
                  inputRef={emailRef}
                  error={errors.email}
                />

                {/* Username */}
                <div style={{ width: 350, textAlign: "left", fontSize: 13 }}>
                  <label
                    style={{
                      display: "block",
                      marginBottom: 6,
                      color: "#0d9488",
                      fontWeight: 600
                    }}
                  >
                    Choose a username
                  </label>

                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <input
                      value={username}
                      onChange={(e) =>
                        setUsername(e.target.value.replace(/[^a-zA-Z0-9._]/g, ""))
                      }
                      style={{
                        flex: 1,
                        padding: "10px 12px",
                        borderRadius: 10,
                        border: errors.username
                          ? "2px solid red"
                          : "2px solid #0d9488",
                        outline: "none",
                        fontFamily: "Helvetica"
                      }}
                    />

                    <button
                      onClick={async () => {
                        const ok = await checkUsernameAvailability(username);
                        setErrors((p) => ({ ...p, username: !ok }));
                        alert(ok ? "Username available" : "Username taken");
                      }}
                      style={{
                        padding: "10px 14px",
                        borderRadius: 10,
                        border: "none",
                        background: "#0d9488",
                        color: "#fff",
                        fontWeight: 700,
                        fontFamily: "Helvetica",
                        cursor: "pointer"
                      }}
                    >
                      Check
                    </button>
                  </div>
                </div>

                {/* OAuth */}
                <div style={{ width: "100%" }}>
                  <div
                    style={{
                      textAlign: "center",
                      margin: "12px 0",
                      fontWeight: 600,
                      color: "#666"
                    }}
                  >
                    Or continue with
                  </div>
                  <OAuthButtonsRow size="280px" />
                </div>

                {/* Invite Code */}
                <div style={{ width: 350, textAlign: "center" }}>
                  <button
                    onClick={() => setInviteVisible((v) => !v)}
                    style={{
                      background: "transparent",
                      border: "none",
                      color: "#0d9488",
                      fontWeight: 700,
                      cursor: "pointer",
                      padding: 0
                    }}
                  >
                    {inviteCodeVisible ? "Hide invite code" : "Have an invite code?"}
                  </button>

                  {inviteCodeVisible && (
                    <input
                      value={inviteCode}
                      onChange={(e) => setInviteCode(e.target.value)}
                      placeholder="Enter invite code"
                      style={{
                        marginTop: 8,
                        width: "100%",
                        padding: "10px 12px",
                        borderRadius: 10,
                        border: "2px solid #e6e6e6",
                        outline: "none",
                        fontFamily: "Helvetica"
                      }}
                    />
                  )}
                </div>

                {/* Buttons */}
                <div style={{ display: "flex", gap: 12 }}>
                  <SmallGhostBtn
                    onClick={() => {
                      setGuestMode(true);
                      navigate("/");
                    }}
                  >
                    Try Guest Mode
                  </SmallGhostBtn>
                  <ActionButton text="Next →" onClick={next} />
                </div>
              </>
            )}

            {/* ======================= */}
            {/*           STEP 2        */}
            {/* ======================= */}
            {step === 2 && (
              <>
                <div style={{ width: 350 }}>
                  <label
                    style={{
                      display: "block",
                      fontWeight: 700,
                      marginBottom: 8,
                      color: "#0d9488"
                    }}
                  >
                    Create a password
                  </label>

                  <div style={{ position: "relative" }}>
                    <input
                      ref={passwordRef}
                      type={showPassword ? "text" : "password"}
                      value={password}
                      onChange={(e) => {
                        setPassword(e.target.value);
                        setErrors((p) => ({ ...p, password: false }));
                      }}
                      style={{
                        width: "100%",
                        padding: "14px 12px",
                        borderRadius: 10,
                        border: errors.password
                          ? "2px solid red"
                          : "2px solid #0d9488",
                        outline: "none",
                        fontFamily: "Helvetica"
                      }}
                    />

                    <span
                      onClick={() => setShowPassword((s) => !s)}
                      style={{
                        position: "absolute",
                        right: 10,
                        top: "50%",
                        transform: "translateY(-50%)",
                        cursor: "pointer",
                        color: "#0d9488",
                        userSelect: "none",
                        fontSize: 18
                      }}
                    >
                      {showPassword ? "👁️" : "👁️‍🗨️"}
                    </span>
                  </div>
                </div>

                <div style={{ width: 350, textAlign: "left" }}>
                  <PasswordRules password={password} rules={passwordRules} />
                </div>

                <FloatingInput
                  label="Confirm password"
                  type="password"
                  value={confirm}
                  onChange={(v) => setConfirm(v)}
                  inputRef={confirmRef}
                  error={errors.confirm}
                />

                <div style={{ display: "flex", gap: 12 }}>
                  <SmallGhostBtn onClick={back}>Back</SmallGhostBtn>
                  <ActionButton text="Next →" onClick={next} />
                </div>
              </>
            )}

            {/* ======================= */}
            {/*           STEP 3        */}
            {/* ======================= */}
            {step === 3 && (
              <>
                {/* Names */}
                <div style={{ width: 350, display: "flex", gap: 12 }}>
                  <div style={{ flex: 1 }}>
                    <FloatingInput
                      label="First name"
                      type="text"
                      value={firstName}
                      onChange={(v) => setFirst(v)}
                      inputRef={firstNameRef}
                      error={errors.firstName}
                    />
                  </div>
                  <div style={{ flex: 1 }}>
                    <FloatingInput
                      label="Last name"
                      type="text"
                      value={lastName}
                      onChange={(v) => setLast(v)}
                      error={errors.lastName}
                    />
                  </div>
                </div>

                {/* Avatar */}
                <div style={{ width: 350, textAlign: "left" }}>
                  <label style={{ fontWeight: 700, color: "#0d9488" }}>
                    Pick an avatar
                  </label>

                  <div
                    style={{
                      display: "flex",
                      gap: 12,
                      marginTop: 10,
                      alignItems: "center"
                    }}
                  >
                    <div
                      onClick={() => setAvatarType("initials")}
                      style={avatarCardStyle(avatarType === "initials")}
                    >
                      <div style={{ fontSize: 18, fontWeight: 700 }}>
                        {(firstName[0] || "P") + (lastName[0] || "")}
                      </div>
                    </div>

                    <div
                      onClick={() => setAvatarType("emoji")}
                      style={avatarCardStyle(avatarType === "emoji")}
                    >
                      <div style={{ fontSize: 22 }}>{emoji}</div>
                    </div>

                    <div style={avatarCardStyle(avatarType === "upload")}>
                      <label style={{ cursor: "pointer" }}>
                        <input
                          type="file"
                          accept="image/*"
                          style={{ display: "none" }}
                          onChange={(e) =>
                            e.target.files?.[0] &&
                            handleAvatarUpload(e.target.files[0])
                          }
                        />
                        <div
                          onClick={() => setAvatarType("upload")}
                          style={{
                            padding: 6,
                            fontSize: 14,
                            color: "#0d9488"
                          }}
                        >
                          Upload
                        </div>
                      </label>
                    </div>

                    {uploadedAvatar && (
                      <img
                        src={uploadedAvatar}
                        alt="avatar"
                        style={{
                          width: 48,
                          height: 48,
                          borderRadius: 10,
                          objectFit: "cover"
                        }}
                      />
                    )}
                  </div>
                </div>

                {/* Referral */}
                <div style={{ width: 350, textAlign: "left" }}>
                  <label
                    style={{
                      display: "block",
                      marginBottom: 6,
                      color: "#0d9488",
                      fontWeight: 600
                    }}
                  >
                    Where did you hear about us?
                  </label>

                  <div style={{ display: "flex", gap: 8, flexWrap: "nowrap" }}>
                    {["LinkedIn", "YouTube", "Friend", "GitHub", "Other"].map(
                      (r) => (
                        <button
                          key={r}
                          onClick={() => setReferral(r)}
                          style={{
                            padding: "8px 12px",
                            borderRadius: 999,
                            border:
                              referral === r
                                ? "2px solid #0d9488"
                                : "2px solid #e6e6e6",
                            background:
                              referral === r ? "#0d9488" : "#fff",
                            color: referral === r ? "#fff" : "#0d9488",
                            fontWeight: 700,
                            cursor: "pointer"
                          }}
                        >
                          {r}
                        </button>
                      )
                    )}
                  </div>

                  <div
                    style={{
                      marginTop: 10,
                      display: "flex",
                      alignItems: "center",
                      gap: 10
                    }}
                  >
                    <label
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        cursor: "pointer"
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={newsletter}
                        onChange={() => setNewsletter((s) => !s)}
                      />
                      <span style={{ fontWeight: 600 }}>
                        Subscribe to newsletter
                      </span>
                    </label>
                  </div>

                  <div style={{ marginTop: 12 }}>
                    <label
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={agreeTerms}
                        onChange={() => setShowTermsModal(true)}
                      />
                      <span style={{ fontWeight: 600 }}>
                        I agree to the Terms & Privacy
                      </span>
                    </label>
                  </div>

                  <div
                    style={{
                      marginTop: 8,
                      color: "#666",
                      fontSize: 13
                    }}
                  >
                    <small>
                      Detected timezone: {clientInfo.tz} • {clientInfo.locale} •{" "}
                      {clientInfo.browser} on {clientInfo.os}
                    </small>
                  </div>
                </div>

                <div style={{ display: "flex", gap: 12 }}>
                  <SmallGhostBtn onClick={back}>Back</SmallGhostBtn>
                  <ActionButton text="Next →" onClick={next} />
                </div>
              </>
            )}

            {/* ======================= */}
            {/*           STEP 4        */}
            {/* ======================= */}
            {step === 4 && (
              <>
                <div style={{ width: 350 }}>
                  <label
                    style={{
                      display: "block",
                      marginBottom: 6,
                      color: "#0d9488",
                      fontWeight: 700
                    }}
                  >
                    Security question (optional)
                  </label>

                  <select
                    value={securityQuestion}
                    onChange={(e) => setSecurityQuestion(e.target.value)}
                    style={{ width: "100%", padding: 10, borderRadius: 10 }}
                  >
                    <option value="">Pick a question...</option>
                    <option>What was your first school?</option>
                    <option>What is your mother's maiden name?</option>
                    <option>What was your childhood nickname?</option>
                  </select>

                  {securityQuestion && (
                    <input
                      value={securityAnswer}
                      onChange={(e) => setSecurityAnswer(e.target.value)}
                      placeholder="Answer"
                      style={{
                        marginTop: 8,
                        width: "100%",
                        padding: 10,
                        borderRadius: 10
                      }}
                    />
                  )}
                </div>

                {/* Onboarding checklist */}
                <div
                  style={{
                    width: 350,
                    textAlign: "left",
                    marginTop: 6
                  }}
                >
                  <h4 style={{ margin: "8px 0", color: "#0d9488" }}>
                    Onboarding checklist
                  </h4>

                  <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
                    {Object.entries(checklist).map(([k, v]) => (
                      <li
                        key={k}
                        style={{ marginBottom: 6 }}
                      >
                        <span
                          style={{
                            fontWeight: 700,
                            color: v ? "#0d9488" : "#999"
                          }}
                        >
                          {v ? "✔" : "●"}
                        </span>{" "}
                        <span
                          style={{
                            marginLeft: 8,
                            color: "#333"
                          }}
                        >
                          {k}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div style={{ display: "flex", gap: 12 }}>
                  <SmallGhostBtn onClick={back}>Back</SmallGhostBtn>
                  <ActionButton
                    text="Create Account"
                    onClick={submitSignup}
                  />
                </div>
              </>
            )}
          </motion.div>
        </motion.div>
      </div>

      {/* Terms Modal */}
      {showTermsModal && (
        <TermsModal
          onClose={() => {
            setShowTermsModal(false);
            setAgreeTerms(true);
            setTermsScrolledToBottom(true);
          }}
          onScrollBottom={() => setTermsScrolledToBottom(true)}
        />
      )}
    </MainLayout>
  );
}

/* ===========================
   UI HELPER COMPONENTS
   =========================== */

function FloatingInput({ label, value, onChange, error, type = "text", inputRef }) {
  return (
    <div style={{ position: "relative", width: 350 }}>
      <input
        ref={inputRef}
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: "100%",
          padding: "16px 12px",
          borderRadius: 10,
          border: error ? "2px solid red" : "2px solid #e6e6e6",
          outline: "none",
          fontFamily: "Helvetica",
          fontSize: 15
        }}
      />

      <label
        style={{
          position: "absolute",
          left: 14,
          top: value ? 6 : 16,
          fontSize: value ? 12 : 15,
          color: error ? "red" : "#0d9488",
          transition: "all 0.15s ease",
          background: "#fff",
          padding: "0 6px"
        }}
      >
        {label}
      </label>

      {error && typeof error === "string" && (
        <div style={{ color: "red", marginTop: 8 }}>{error}</div>
      )}
    </div>
  );
}

function PasswordRules({ password, rules }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {rules.map((r) => {
        const ok = r.test(password);
        return (
          <div
            key={r.key}
            style={{
              display: "flex",
              gap: 8,
              alignItems: "center",
              color: ok ? "#0d9488" : "#999",
              fontWeight: ok ? 700 : 500
            }}
          >
            <div style={{ width: 16 }}>{ok ? "✔" : "●"}</div>
            <div>{r.label}</div>
          </div>
        );
      })}
    </div>
  );
}

function ActionButton({ text, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        background: "#0d9488",
        color: "#fff",
        padding: "12px 18px",
        borderRadius: 999,
        border: "none",
        fontWeight: 700,
        cursor: "pointer",
        fontFamily: "Helvetica"
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = "#fff";
        e.currentTarget.style.color = "#0d9488";
        e.currentTarget.style.border = "2px solid #0d9488";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = "#0d9488";
        e.currentTarget.style.color = "#fff";
        e.currentTarget.style.border = "none";
      }}
    >
      {text}
    </button>
  );
}

function SmallGhostBtn({ children, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "10px 14px",
        borderRadius: 10,
        border: "1px solid #e6e6e6",
        background: "transparent",
        cursor: "pointer",
        fontWeight: 700,
        color: "#0d9488",
        fontFamily: "Helvetica"
      }}
    >
      {children}
    </button>
  );
}

function TermsModal({ onClose, onScrollBottom }) {
  const contentRef = useRef(null);

  useEffect(() => {
    const el = contentRef.current;
    if (!el) return;

    const onScroll = () => {
      if (el.scrollTop + el.clientHeight >= el.scrollHeight - 6) {
        onScrollBottom();
      }
    };

    el.addEventListener("scroll", onScroll);
    return () => el.removeEventListener("scroll", onScroll);
  }, [onScrollBottom]);

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.5)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 2000
      }}
    >
      <div
        style={{
          width: "80%",
          maxWidth: 900,
          background: "#fff",
          borderRadius: 10,
          padding: 20
        }}
      >
        <h2 style={{ fontFamily: "Helvetica", color: "#0d9488" }}>
          Terms & Privacy
        </h2>

        <div
          ref={contentRef}
          style={{
            maxHeight: 320,
            overflow: "auto",
            padding: 8,
            border: "1px solid #eee"
          }}
        >
          <p style={{ lineHeight: 1.6 }}>
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer
            eget augue at lacus rhoncus ullamcorper... (Scroll to bottom to
            agree)
          </p>

          <p style={{ lineHeight: 1.6 }}>
            Pellentesque habitant morbi tristique senectus et netus et malesuada
            fames ac turpis egestas...
          </p>

          <div style={{ height: 400 }} />
        </div>

        <div
          style={{
            display: "flex",
            gap: 12,
            justifyContent: "flex-end",
            marginTop: 12
          }}
        >
          <SmallGhostBtn onClick={onClose}>Agree & Close</SmallGhostBtn>
        </div>
      </div>
    </div>
  );
}

/* =========================
   Styles helpers
   ========================= */

function avatarCardStyle(active) {
  return {
    width: 56,
    height: 56,
    borderRadius: 10,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    border: active ? "2px solid #0d9488" : "2px solid #e6e6e6",
    cursor: "pointer",
    background: "#fff"
  };
}
