/** Browser text to speech. Never let a speech failure break the demo. */
export function speak(text) {
  if (!text) return
  try {
    const u = new SpeechSynthesisUtterance(text)
    u.lang = 'en-IN'
    window.speechSynthesis.cancel()
    window.speechSynthesis.speak(u)
  } catch {
    /* speech is a bonus, not a requirement */
  }
}
