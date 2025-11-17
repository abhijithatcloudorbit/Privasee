// passport-config.js
const passport = require("passport");
const GoogleStrategy = require("passport-google-oauth20").Strategy;
const GitHubStrategy = require("passport-github2").Strategy;
const MicrosoftStrategy = require("passport-microsoft").Strategy;
require("dotenv").config();

module.exports = function configurePassport() {
  // Serialize / deserialize (we'll not use sessions heavily, but required)
  passport.serializeUser((user, done) => done(null, user));
  passport.deserializeUser((obj, done) => done(null, obj));

  // Google Strategy
  passport.use(
    new GoogleStrategy(
      {
        clientID: process.env.GOOGLE_CLIENT_ID,
        clientSecret: process.env.GOOGLE_CLIENT_SECRET,
        callbackURL: `${process.env.BASE_URL}${process.env.GOOGLE_CALLBACK}`,
      },
      function (accessToken, refreshToken, profile, done) {
        // Normalise user object
        const user = {
          id: profile.id,
          provider: "google",
          displayName: profile.displayName,
          emails: profile.emails,
          photos: profile.photos,
        };
        return done(null, user);
      }
    )
  );

  // GitHub Strategy
  passport.use(
    new GitHubStrategy(
      {
        clientID: process.env.GITHUB_CLIENT_ID,
        clientSecret: process.env.GITHUB_CLIENT_SECRET,
        callbackURL: `${process.env.BASE_URL}${process.env.GITHUB_CALLBACK}`,
        scope: ["user:email"],
      },
      function (accessToken, refreshToken, profile, done) {
        const user = {
          id: profile.id,
          provider: "github",
          displayName: profile.displayName || profile.username,
          emails: profile.emails,
          photos: profile.photos,
        };
        return done(null, user);
      }
    )
  );

  // Microsoft Strategy (Outlook)
  passport.use(
    new MicrosoftStrategy(
      {
        clientID: process.env.MICROSOFT_CLIENT_ID,
        clientSecret: process.env.MICROSOFT_CLIENT_SECRET,
        callbackURL: `${process.env.BASE_URL}${process.env.MICROSOFT_CALLBACK}`,
        scope: ["user.read"],
      },
      function (accessToken, refreshToken, profile, done) {
        const user = {
          id: profile.id,
          provider: "microsoft",
          displayName: profile.displayName,
          emails: profile.emails,
          photos: profile.photos,
        };
        return done(null, user);
      }
    )
  );

  // === Apple setup note ===
  // Apple Sign In requires a signed JWT client secret and a private key (.p8).
  // You can integrate with passport-apple or apple-signin-auth.
  // I did not wire a strategy here by default — see README below for Apple example.
};
