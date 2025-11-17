import express from "express";
import jwt from "jsonwebtoken";
import passport from "passport";

const router = express.Router();

function sendToken(req, res) {
  const user = req.user;

  const token = jwt.sign(
    { id: user.id, email: user.email, provider: user.provider },
    process.env.JWT_SECRET,
    { expiresIn: process.env.JWT_EXPIRES_IN }
  );

  return res.redirect(
    `${process.env.FRONTEND_URL}/?token=${token}&user=${encodeURIComponent(
      user.name
    )}`
  );
}

/* ===== GOOGLE ===== */
router.get("/google", passport.authenticate("google", { scope: ["profile", "email"] }));
router.get(
  "/google/callback",
  passport.authenticate("google", { failureRedirect: "/auth/fail" }),
  sendToken
);

/* ===== GITHUB ===== */
router.get("/github", passport.authenticate("github", { scope: ["user:email"] }));
router.get(
  "/github/callback",
  passport.authenticate("github", { failureRedirect: "/auth/fail" }),
  sendToken
);

/* ===== MICROSOFT ===== */
router.get(
  "/microsoft",
  passport.authenticate("microsoft", { scope: ["user.read"] })
);
router.get(
  "/microsoft/callback",
  passport.authenticate("microsoft", { failureRedirect: "/auth/fail" }),
  sendToken
);

/* ===== APPLE ===== */
router.get("/apple", passport.authenticate("apple"));
router.post(
  "/apple/callback",
  passport.authenticate("apple", { failureRedirect: "/auth/fail" }),
  sendToken
);

router.get("/fail", (req, res) => res.send("OAuth Login Failed"));

export default router;
