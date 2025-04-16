import Link from "next/link"

export default function Home() {
  return (
    <div className="bg-black bg-[url(/images/home-img-bg.jpeg)] bg-cover bg-center">
      <main className="flex flex-col justify-center text-center max-w-5xl mx-auto h-dvh">

        <div className="flex flex-col gap-6 p-12 rounded-xl bg-black/90 w-4/5 sm:max-w-96 mx-auto text-white sm:text-2xl">
          <h1 className="text-4xl font-bold">Vintage Planner<br/>Plan Volumes</h1>
          <p>Flavor text that is an hourly callout in example</p>
          <Link href="tel:5555555555"
          className="hover:underline">
            555-555-5555
          </Link>
        </div>

      </main>

    </div>
  );
}