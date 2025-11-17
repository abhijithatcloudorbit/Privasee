import passport from "passport";
import GoogleStrategy from "passport-google-oauth20";
import GitHubStrategy from "passport-github2";
import MicrosoftStrategy from "passport-microsoft";
import fs from "fs";
import path from "path";
import dotenv from "dotenv";

dotenv.config();

// ===== Helper: After login, we store user info =====
function authCallback(accessToken, refreshToken, profile, done) {
  const user = {
    id: profile.id,
    name: profile.displayName,
    provider: profile.provider,
    email: profile.emails ? profile.emails[0].value : null,
  };
  return done(null, user);
}

export default function configurePassport() {
  // ----- GOOGLE -----
  passport.use(
    new GoogleStrategy(
      {
        clientID: process.env.GOOGLE_CLIENT_ID,
        clientSecret: process.env.GOOGLE_CLIENT_SECRET,
        callbackURL: process.env.GOOGLE_CALLBACK,
      },
      authCallback
    )
  );

  // ----- GITHUB -----
  passport.use(
    new GitHubStrategy(
      {
        clientID: process.env.GITHUB_CLIENT_ID,
        clientSecret: process.env.GITHUB_CLIENT_SECRET,
        callbackURL: process.env.GITHUB_CALLBACK,
      },
      authCallback
    )
  );

  // ----- MICROSOFT / OUTLOOK -----
  passport.use(
    new MicrosoftStrategy(
      {
        clientID: process.env.MICROSOFT_CLIENT_ID,
        clientSecret: process.env.MICROSOFT_CLIENT_SECRET,
        callbackURL: process.env.MICROSOFT_CALLBACK,
        tenant: process.env.MICROSOFT_TENANT,
      },
      authCallback
    )
  );

  // ----- APPLE (optional) -----
  try {
    const privateKey = fs.readFileSync(
      path.resolve(process.env.APPLE_PRIVATE_KEY_PATH)
    );

    passport.use(
      new AppleStrategy(
        {
          clientID: process.env.APPLE_CLIENT_ID,
          teamID: process.env.APPLE_TEAM_ID,
          keyID: process.env.APPLE_KEY_ID,
          privateKey,
          callbackURL: process.env.APPLE_CALLBACK,
          scope: ["name", "email"],
        },
        authCallback
      )
    );
  } catch (e) {
    console.log("Apple key not found, skipping Apple login.");
  }

  passport.serializeUser((user, done) => done(null, user));
  passport.deserializeUser((obj, done) => done(null, obj));
}
