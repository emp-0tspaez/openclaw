import { describe, expect, test } from "vitest";
import { stripInboundMeta, stripEnvelopeFromMessage } from "./chat-sanitize.js";

describe("stripInboundMeta", () => {
  test("strips conversation info block", () => {
    const input = [
      'Conversation info (untrusted metadata):',
      '```json',
      '{',
      '  "conversation_label": "A34 de Salvador"',
      '}',
      '```',
      '',
      'hello',
    ].join("\n");
    expect(stripInboundMeta(input)).toBe("hello");
  });

  test("strips sender info block", () => {
    const input = [
      'Sender (untrusted metadata):',
      '```json',
      '{ "label": "Hugo" }',
      '```',
      '',
      'hi',
    ].join("\n");
    expect(stripInboundMeta(input)).toBe("hi");
  });

  test("strips chat history block", () => {
    const input = [
      'Chat history since last reply (untrusted, for context):',
      '```json',
      '[{"sender":"Hugo","body":"test"}]',
      '```',
      '',
      'yo',
    ].join("\n");
    expect(stripInboundMeta(input)).toBe("yo");
  });

  test("strips system event lines", () => {
    const input = "System: [2026-02-14 22:18:05 CST] Node: A34 connected\nhello";
    expect(stripInboundMeta(input)).toBe("hello");
  });

  test("strips Talk Mode prefix", () => {
    const input = [
      'Talk Mode active. Reply in a concise, spoken tone.',
      'You may optionally prefix the response with JSON (first line) to set ElevenLabs voice (id or alias), e.g. {"voice":"<id>","once":true}.',
      'moro',
    ].join("\n");
    expect(stripInboundMeta(input)).toBe("moro");
  });

  test("strips multiple metadata blocks at once", () => {
    const input = [
      'Conversation info (untrusted metadata):',
      '```json',
      '{ "conversation_label": "A34 de Salvador" }',
      '```',
      '',
      'Sender (untrusted metadata):',
      '```json',
      '{ "label": "Hugo" }',
      '```',
      '',
      'System: [2026-02-14 22:18:05 CST] Node: A34 connected',
      '',
      'Talk Mode active. Reply in a concise, spoken tone.',
      'You may optionally prefix the response with JSON (first line) to set ElevenLabs voice.',
      'hola mundo',
    ].join("\n");
    expect(stripInboundMeta(input)).toBe("hola mundo");
  });

  test("strips abort hint", () => {
    const input = "Note: The previous agent run was aborted by the user. Resume carefully or ask for clarification.\n\nhello";
    expect(stripInboundMeta(input)).toBe("hello");
  });

  test("strips interrupted marker", () => {
    const input = "[interrupted at 3.2s]\nhello";
    expect(stripInboundMeta(input)).toBe("hello");
  });

  test("strips media placeholder", () => {
    const input = "[User sent media without caption]\nhello";
    expect(stripInboundMeta(input)).toBe("hello");
  });

  test("does not strip normal user text", () => {
    expect(stripInboundMeta("hello world")).toBe("hello world");
  });

  test("returns empty string for metadata-only messages", () => {
    const input = [
      'Conversation info (untrusted metadata):',
      '```json',
      '{ "conversation_label": "Test" }',
      '```',
    ].join("\n");
    expect(stripInboundMeta(input)).toBe("");
  });

  test("strips thread context blocks", () => {
    const input = "[Thread history - for context]\nsome old message\n\nhello";
    expect(stripInboundMeta(input)).toBe("hello");
  });

  test("strips replied message block", () => {
    const input = [
      'Replied message (untrusted, for context):',
      '```json',
      '{ "sender_label": "Hugo", "body": "hey" }',
      '```',
      '',
      'sup',
    ].join("\n");
    expect(stripInboundMeta(input)).toBe("sup");
  });

  test("strips forwarded message block", () => {
    const input = [
      'Forwarded message context (untrusted metadata):',
      '```json',
      '{ "from": "Hugo" }',
      '```',
      '',
      'check this',
    ].join("\n");
    expect(stripInboundMeta(input)).toBe("check this");
  });
});

describe("stripEnvelopeFromMessage strips metadata end-to-end", () => {
  test("full session message is cleaned", () => {
    const rawContent = [
      'Conversation info (untrusted metadata):',
      '```json',
      '{',
      '  "conversation_label": "A34 de Salvador"',
      '}',
      '```',
      '',
      '[Sat 2026-02-14 22:18 CST] Talk Mode active. Reply in a concise, spoken tone.',
      'You may optionally prefix the response with JSON (first line) to set ElevenLabs voice (id or alias), e.g. {"voice":"<id>","once":true}.',
      'moro',
    ].join("\n");
    const input = {
      role: "user",
      content: [{ type: "text", text: rawContent }],
    };
    const result = stripEnvelopeFromMessage(input) as {
      content?: Array<{ type: string; text?: string }>;
    };
    expect(result.content?.[0]?.text).toBe("moro");
  });
});

describe("stripEnvelopeFromMessage", () => {
  test("removes message_id hint lines from user messages", () => {
    const input = {
      role: "user",
      content: "[WhatsApp 2026-01-24 13:36] yolo\n[message_id: 7b8b]",
    };
    const result = stripEnvelopeFromMessage(input) as { content?: string };
    expect(result.content).toBe("yolo");
  });

  test("removes message_id hint lines from text content arrays", () => {
    const input = {
      role: "user",
      content: [{ type: "text", text: "hi\n[message_id: abc123]" }],
    };
    const result = stripEnvelopeFromMessage(input) as {
      content?: Array<{ type: string; text?: string }>;
    };
    expect(result.content?.[0]?.text).toBe("hi");
  });

  test("does not strip inline message_id text that is part of a line", () => {
    const input = {
      role: "user",
      content: "I typed [message_id: 123] on purpose",
    };
    const result = stripEnvelopeFromMessage(input) as { content?: string };
    expect(result.content).toBe("I typed [message_id: 123] on purpose");
  });

  test("does not strip assistant messages", () => {
    const input = {
      role: "assistant",
      content: "note\n[message_id: 123]",
    };
    const result = stripEnvelopeFromMessage(input) as { content?: string };
    expect(result.content).toBe("note\n[message_id: 123]");
  });
});
