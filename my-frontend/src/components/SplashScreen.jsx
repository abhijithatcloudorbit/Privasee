import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Typed from "typed.js";

export default function SplashScreen({ onFinish }) {
  const typingRef = useRef(null);

  useEffect(() => {
    const typed = new Typed(typingRef.current, {
      strings: ["Privasee"],
      typeSpeed: 140,
      showCursor: false,
      startDelay: 300,

      // 🚀 When typing finishes → fade out splash
      onComplete: () => {
        setTimeout(() => {
          onFinish();
        }, 300); // small delay for smooth fade-out
      },
    });

    return () => typed.destroy();
  }, []);

  return (
    <AnimatePresence>
      <motion.div
        className="splash-container"
        initial={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.6 }}
      >
        <motion.div
          initial={{ scale: 0.6, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <h1
            ref={typingRef}
            style={{
              fontSize: "10rem",
              fontWeight: "700",
              textAlign: "center",
              fontFamily: "Helvetica, Arial, sans-serif",
              color: "#ffffff",
              letterSpacing: "-2px",
            }}
          ></h1>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
