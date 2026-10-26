/* Confidant demo behaviour, loaded once in <head>.
   Listeners are delegated from document so they survive Gradio re-renders.
   - [data-prompt]  (demo script): put the prompt in the composer and send it.
   - [data-open-tab] (guard line): switch the inspector to that tab.
   - [data-mem]     (chips, citations, filled values, table rows): open the
     provenance dialog, built from the templates rendered in the Memory tab.
   - When the transcript changes, bring the newest turn into view. */
(() => {
  "use strict";

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const scrollBehavior = () => (reducedMotion.matches ? "auto" : "smooth");

  /** Return the provenance dialog, creating it on first use. */
  function provenanceDialog() {
    let dialog = document.getElementById("cf-dialog");
    if (dialog) return dialog;
    dialog = document.createElement("dialog");
    dialog.id = "cf-dialog";
    dialog.className = "cf-dialog";
    dialog.setAttribute("aria-labelledby", "cf-dialog-title");
    dialog.innerHTML =
      '<form method="dialog"><button class="cf-dialog__close" aria-label="Close">&times;</button></form>' +
      '<div class="cf-dialog__body"></div>';
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close(); // click on the backdrop
    });
    document.body.appendChild(dialog);
    return dialog;
  }

  /** Show provenance and version history for memory `id`; highlight version `version`. */
  function openMemory(id, version) {
    const template = document.querySelector(
      `#cf-mem-details template[data-mem-detail="${CSS.escape(id)}"]`
    );
    if (!template) return;
    const dialog = provenanceDialog();
    const body = dialog.querySelector(".cf-dialog__body");
    body.innerHTML = template.innerHTML;
    const cited = version && body.querySelector(`.cf-history li[data-ver="${CSS.escape(version)}"]`);
    if (cited) {
      cited.classList.add("is-cited");
      const note = document.createElement("span");
      note.className = "cf-cited-note";
      note.textContent = "The version cited here";
      cited.prepend(note);
    }
    if (!dialog.open) dialog.showModal();
    body.scrollTop = 0;
  }

  /** Type `text` into the composer and press Send, as a visitor would. */
  function sendPrompt(text) {
    const box = document.querySelector("#cf-input textarea");
    const send = document.getElementById("cf-send");
    if (!box || !send) return;
    box.value = text;
    box.dispatchEvent(new Event("input", { bubbles: true }));
    window.requestAnimationFrame(() => send.click());
  }

  /** Select an inspector tab by name and scroll it into view on small screens. */
  function openTab(name) {
    const tabButton = document.getElementById(`cf-tab-${name}-button`);
    if (!tabButton) return;
    tabButton.click();
    const inspector = document.getElementById("cf-inspector-col");
    if (inspector && inspector.getBoundingClientRect().top > window.innerHeight * 0.6) {
      inspector.scrollIntoView({ behavior: scrollBehavior(), block: "start" });
    }
  }

  document.addEventListener("click", (event) => {
    const prompt = event.target.closest("[data-prompt]");
    if (prompt) {
      sendPrompt(prompt.dataset.prompt);
      return;
    }
    const tab = event.target.closest("[data-open-tab]");
    if (tab) {
      openTab(tab.dataset.openTab);
      return;
    }
    const memory = event.target.closest("[data-mem]");
    if (memory) openMemory(memory.dataset.mem, memory.dataset.ver);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const memory = event.target.closest && event.target.closest("[data-mem]:not(button)");
    if (!memory) return;
    event.preventDefault();
    openMemory(memory.dataset.mem, memory.dataset.ver);
  });

  /** Scroll the newest turn to the top of the transcript (or of the page on phones). */
  function revealLatestTurn(transcript) {
    const turns = transcript.querySelectorAll(".cf-turn");
    const latest = turns[turns.length - 1];
    if (!latest) return;
    const scrollsItself = getComputedStyle(transcript).overflowY !== "visible";
    if (scrollsItself) {
      const offset = latest.getBoundingClientRect().top - transcript.getBoundingClientRect().top;
      transcript.scrollTo({ top: transcript.scrollTop + offset - 16, behavior: scrollBehavior() });
    } else {
      latest.scrollIntoView({ behavior: scrollBehavior(), block: "start" });
    }
  }

  /** Watch the transcript once Gradio has mounted it. */
  function watchTranscript() {
    const transcript = document.getElementById("cf-transcript");
    if (!transcript) return false;
    let lastSignature = "";
    new MutationObserver(() => {
      const turns = transcript.querySelectorAll(".cf-turn");
      const signature = `${turns.length}:${transcript.querySelectorAll(".cf-pending").length}`;
      if (signature === lastSignature) return;
      lastSignature = signature;
      revealLatestTurn(transcript);
    }).observe(transcript, { childList: true, subtree: true });
    return true;
  }

  if (!watchTranscript()) {
    const mountWatcher = new MutationObserver(() => {
      if (watchTranscript()) mountWatcher.disconnect();
    });
    mountWatcher.observe(document.documentElement, { childList: true, subtree: true });
  }
})();
