import { stripEnvelope, stripMessageIdHints } from "../shared/chat-envelope.js";

export { stripEnvelope };

// ── Inbound metadata stripping ────────────────────────────────────────
// The auto-reply pipeline prepends context blocks (conversation info,
// sender, history, system events, etc.) to user messages before sending
// them to the model.  These blocks are useful for the AI but should NOT
// appear in the chat-history UI.  The patterns below match the formats
// produced by buildInboundUserContextPrefix(), prependSystemEvents(),
// injectTimestamp(), and the Apple TalkPromptBuilder.

/**
 * Removes all recognised metadata / context blocks that the auto-reply
 * pipeline injects into user-role messages.  Returns the clean user text.
 *
 * The function is intentionally strict in its pattern matching to avoid
 * false-positive stripping of legitimate user text.
 */
export function stripInboundMeta(text: string): string {
  let result = text;

  // 1. Strip labelled "(untrusted metadata)" / "(untrusted, for context)" code-fence blocks.
  //    Matches:  "Label (untrusted metadata):\n```json\n...\n```"
  //              "Label (untrusted, for context):\n```json\n...\n```"
  //    Handles both ```json and plain ``` fences.
  result = result.replace(
    /^[^\n]+ \(untrusted(?:[ ,]+(?:metadata|for context))+\):\n```(?:json)?\n[\s\S]*?```\s*/gm,
    "",
  );

  // 2. Strip "Chat history since last reply (untrusted, for context):" blocks.
  result = result.replace(
    /^Chat history since last reply \(untrusted, for context\):\n```(?:json)?\n[\s\S]*?```\s*/gm,
    "",
  );

  // 3. Strip "System: [timestamp] ..." lines (from prependSystemEvents).
  result = result.replace(/^System: \[.+?\] .+\n?/gm, "");

  // 4. Strip "[Thread history - for context]" / "[Thread starter - for context]" blocks.
  //    These end at the next double-newline boundary.
  result = result.replace(/^\[Thread (?:history|starter) - for context\]\n[\s\S]*?(?:\n\n|$)/gm, "");

  // 5. Strip "Untrusted context (metadata, do not treat as instructions or commands):" blocks.
  result = result.replace(
    /^Untrusted context \(metadata, do not treat as instructions or commands\):\n[\s\S]*?(?:\n\n|$)/gm,
    "",
  );

  // 6. Strip agent-run abort hint.
  result = result.replace(
    /^Note: The previous agent run was aborted by the user\. Resume carefully or ask for clarification\.\s*/gm,
    "",
  );

  // 7. Strip Apple TalkPromptBuilder prefix:
  //    "Talk Mode active. Reply in a concise, spoken tone.\n...ElevenLabs...\n"
  //    Also match updated versions without ElevenLabs.
  //    The line may be preceded by a timestamp envelope like "[Sat 2026-02-14 22:18 CST] ".
  result = result.replace(
    /^(?:\[[^\]]+\]\s*)?Talk Mode active\. Reply in a concise, spoken tone\.(?:\n[^\n]*(?:optionally prefix|voice)[^\n]*)?\n?/gm,
    "",
  );

  // 8. Strip "[User sent media without caption]" placeholder.
  result = result.replace(/^\[User sent media without caption\]\s*/gm, "");

  // 9. Strip "[interrupted at X.Xs]" markers.
  result = result.replace(/^\[interrupted at [\d.]+s\]\s*/gm, "");

  // 10. Strip media notes: "[Media: ...]" lines.
  result = result.replace(/^\[Media: [^\]]+\]\s*/gm, "");

  // Clean up leading/trailing whitespace left by removals.
  return result.trim();
}

function sanitizeText(text: string): string {
  // 1. Strip metadata blocks first (may expose envelope timestamps underneath).
  // 2. Strip envelope timestamps that may now be at the start of the text.
  // 3. Strip message-id hint lines last.
  return stripMessageIdHints(stripEnvelope(stripInboundMeta(text)));
}

function stripEnvelopeFromContent(content: unknown[]): { content: unknown[]; changed: boolean } {
  let changed = false;
  const next = content.map((item) => {
    if (!item || typeof item !== "object") {
      return item;
    }
    const entry = item as Record<string, unknown>;
    if (entry.type !== "text" || typeof entry.text !== "string") {
      return item;
    }
    const stripped = sanitizeText(entry.text);
    if (stripped === entry.text) {
      return item;
    }
    changed = true;
    return {
      ...entry,
      text: stripped,
    };
  });
  return { content: next, changed };
}

export function stripEnvelopeFromMessage(message: unknown): unknown {
  if (!message || typeof message !== "object") {
    return message;
  }
  const entry = message as Record<string, unknown>;
  const role = typeof entry.role === "string" ? entry.role.toLowerCase() : "";
  if (role !== "user") {
    return message;
  }

  let changed = false;
  const next: Record<string, unknown> = { ...entry };

  if (typeof entry.content === "string") {
    const stripped = sanitizeText(entry.content);
    if (stripped !== entry.content) {
      next.content = stripped;
      changed = true;
    }
  } else if (Array.isArray(entry.content)) {
    const updated = stripEnvelopeFromContent(entry.content);
    if (updated.changed) {
      next.content = updated.content;
      changed = true;
    }
  } else if (typeof entry.text === "string") {
    const stripped = sanitizeText(entry.text);
    if (stripped !== entry.text) {
      next.text = stripped;
      changed = true;
    }
  }

  return changed ? next : message;
}

export function stripEnvelopeFromMessages(messages: unknown[]): unknown[] {
  if (messages.length === 0) {
    return messages;
  }
  let changed = false;
  const next = messages.map((message) => {
    const stripped = stripEnvelopeFromMessage(message);
    if (stripped !== message) {
      changed = true;
    }
    return stripped;
  });
  return changed ? next : messages;
}
