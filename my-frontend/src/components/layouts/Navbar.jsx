import { motion } from "framer-motion";

export default function Navbar() {
  return (
    <motion.nav
      initial={{ y: -80, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{
        duration: 0.6,
        ease: "easeOut"
      }}
      style={{
        width: "100%",
        padding: "20px 40px",
        backgroundColor: "#0d9488", // teal
        color: "white",
        fontFamily: "Helvetica, Arial, sans-serif",
        fontWeight: 600,
        fontSize: "1.2rem",
        position: "fixed",
        top: 0,
        left: 0,
        zIndex: 1000
      }}
    >
      Privasee
    </motion.nav>
  );
}
