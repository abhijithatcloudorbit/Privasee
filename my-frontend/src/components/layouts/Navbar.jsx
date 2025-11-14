import { motion } from "framer-motion";
import { Link } from "react-router-dom";

export default function Navbar() {
  return (
    <motion.nav
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.7, ease: "easeOut" }}
      style={{
        width: "100%",
        padding: "20px 40px",
        backgroundColor: "#0d9488",
        color: "white",
        fontFamily: "Helvetica, Arial, sans-serif",
        fontWeight: 600,
        fontSize: "2.4rem",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        position: "fixed",
        top: 0,
        left: 0,
        zIndex: 999,
      }}
    >
      {/* LEFT LOGO */}
      <div>Privasee</div>

      {/* RIGHT NAV BUTTONS */}
      <div style={{ display: "flex", gap: "14px", marginRight: "60px" }}>
        {["Home", "About", "Contact", "Dashboard"].map((label) => (
          <motion.div key={label} style={{ borderRadius: "50px" }}>
            <motion.div
              whileHover={{
                backgroundColor: "#ffffff", // only button toggles
                color: "#0d9488",
              }}
              whileTap={{ scale: 0.97 }}
              style={{
                padding: "10px 22px",
                borderRadius: "50px",
                backgroundColor: "#0d9488",
                color: "#ffffff",
                fontSize: "1.25rem",
                fontWeight: "600",
                cursor: "pointer",
                fontFamily: "Helvetica, Arial, sans-serif",
                transition: "all 0.2s ease",
              }}
            >
              <Link
                to={label === "Home" ? "/" : `/${label.toLowerCase()}`}
                style={{
                  textDecoration: "none",
                  color: "inherit",
                }}
              >
                {label}
              </Link>
            </motion.div>
          </motion.div>
        ))}
      </div>
    </motion.nav>
  );
}
