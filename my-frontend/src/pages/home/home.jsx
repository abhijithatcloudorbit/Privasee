import MainLayout from "@/layout/MainLayout.jsx";
import PrimaryButton from "../../components/buttons/primarybutton";

export default function Home() {
  return (
    <MainLayout>
      <h1>Welcome to Privasee</h1>
      <p>Your dashboard starts here.</p>
      <p>Home content...</p>

      <PrimaryButton>Home</PrimaryButton>
    </MainLayout>
  );
}
