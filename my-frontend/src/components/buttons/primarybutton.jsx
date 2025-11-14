import { motion } from "framer-motion";

export default function PrimaryButton({ children, onClick, style = {} }) {
  return (
    <motion.button
      onClick={onClick}
      whileHover={{
        backgroundColor: "#ffffff",
        color: "#0d9488",
        scale: 1.03,
      }}
      whileTap={{ scale: 0.97 }}
      transition={{ duration: 0.2 }}
      style={{
        backgroundColor: "#0d9488",
        color: "#ffffff",
        padding: "12px 28px",
        border: "none",
        borderRadius: "20px",
        fontSize: "1rem",
        fontWeight: "600",
        cursor: "pointer",
        fontFamily: "Helvetica, Arial, sans-serif",
        outline: "none",
        boxShadow: "0 2px 6px rgba(0,0,0,0.15)",
        ...style,
      }}
    >
      {children}
    </motion.button>
  );
}
