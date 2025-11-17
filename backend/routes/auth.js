// routes/auth.js
const express = require("express");
const passport = require("passport");
const jwt = require("jsonwebtoken");
const router = express.Router();
require("dotenv").config();

// helper: create JWT and redirect to frontend
function finishAuth(req, res, user) {
  const payload = {
    sub: user.id,
    name: user.displayName,
    provider: user.provider,
    emails: user.emails || [],
  };

  const token = jwt.sign(payload, process.env.JWT_SECRET, {
    expiresIn: process.env.JWT_EXPIRES_IN || "1h",
  });

  // redirect to frontend with token (use secure method in production)
  const redirectUrl = `${process.env.FRONTEND_URL}/auth/callback?token=${token}`;
  return res.redirect(redirectUrl);
}

/**
 * Google
 */
router.get("/google", passport.authenticate("google", { scope: ["profile", "email"] }));

router.get(
  "/google/callback",
  passport.authenticate("google", { session: false, failureRedirect: `${process.env.FRONTEND_URL}/login?error=google` }),
  (req, res) => {
    finishAuth(req, res, req.user);
  }
);

/**
 * GitHub
 */
router.get("/github", passport.authenticate("github", { scope: ["user:email"] }));

router.get(
  "/github/callback",
  passport.authenticate("github", { session: false, failureRedirect: `${process.env.FRONTEND_URL}/login?error=github` }),
  (req, res) => {
    finishAuth(req, res, req.user);
  }
);

/**
 * Microsoft / Outlook
 */
router.get("/microsoft", passport.authenticate("microsoft"));

router.get(
  "/microsoft/callback",
  passport.authenticate("microsoft", { session: false, failureRedirect: `${process.env.FRONTEND_URL}/login?error=microsoft` }),
  (req, res) => {
    finishAuth(req, res, req.user);
  }
);

/**
 * Optional: Logout
 */
router.get("/logout", (req, res) => {
  req.logout?.();
  res.redirect(process.env.FRONTEND_URL || "/");
});

module.exports = router;
