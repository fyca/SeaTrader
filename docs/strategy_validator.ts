import Ajv, { ErrorObject } from "ajv";
import addFormats from "ajv-formats";
import strategySchema from "./strategy.schema.json";
import prelaunchSchema from "./prelaunch.schema.json";

type Finding = { code: string; path?: string; message: string };
export type ValidationResult = { ok: boolean; errors: Finding[]; warnings: Finding[] };

const ajv = new Ajv({ allErrors: true, strict: false });
addFormats(ajv);

const validateStrategySchema = ajv.compile(strategySchema as object);
const validatePrelaunchSchema = ajv.compile(prelaunchSchema as object);

function mapAjvErrors(errors?: ErrorObject[] | null): Finding[] {
  return (errors ?? []).map((e) => ({
    code: "SCHEMA_ERROR",
    path: e.instancePath || "",
    message: e.message || "schema validation error"
  }));
}

function collectRefs(node: any, out: string[] = []): string[] {
  if (!node || typeof node !== "object") return out;
  if (typeof node.ref === "string") out.push(node.ref);
  if (Array.isArray(node.args)) node.args.forEach((a) => collectRefs(a, out));
  return out;
}

export function validateStrategyDoc(doc: any): ValidationResult {
  const errors: Finding[] = [];
  const warnings: Finding[] = [];

  const okSchema = validateStrategySchema(doc);
  if (!okSchema) errors.push(...mapAjvErrors(validateStrategySchema.errors));
  if (!okSchema) return { ok: false, errors, warnings };

  for (const cls of ["stocks", "crypto"] as const) {
    for (const side of ["entry_policy", "exit_policy"] as const) {
      const policy = doc.asset_policies?.[cls]?.[side];
      const indicators = policy?.indicators ?? [];
      const rules = policy?.rules ?? [];

      // Cross-field indicator checks
      indicators.forEach((ind: any, i: number) => {
        if (ind.indicator === "macd") {
          const { fast, slow } = ind.params ?? {};
          if (!(Number.isFinite(fast) && Number.isFinite(slow) && fast < slow)) {
            errors.push({
              code: "INVALID_PARAM",
              path: `/asset_policies/${cls}/${side}/indicators/${i}/params`,
              message: "macd.fast must be < macd.slow"
            });
          }
        }
      });

      // Unique priorities
      const seenPri = new Set<number>();
      rules.forEach((r: any, i: number) => {
        if (seenPri.has(r.priority)) {
          errors.push({
            code: "DUPLICATE_PRIORITY",
            path: `/asset_policies/${cls}/${side}/rules/${i}/priority`,
            message: `duplicate priority ${r.priority}`
          });
        }
        seenPri.add(r.priority);
      });

      // Ref resolution for indicator refs
      const keySet = new Set<string>((indicators ?? []).map((x: any) => x.key));
      const re = /^ind\.([a-z][a-z0-9_]{2,63})\.[a-z_]+$/;
      rules.forEach((r: any, i: number) => {
        const refs = collectRefs(r.when);
        refs.forEach((ref) => {
          const m = ref.match(re);
          if (m && !keySet.has(m[1])) {
            errors.push({
              code: "UNRESOLVED_REF",
              path: `/asset_policies/${cls}/${side}/rules/${i}/when`,
              message: `unresolved indicator ref: ${ref}`
            });
          }
        });
      });

      // Action payload checks
      rules.forEach((r: any, i: number) => {
        if (r.action === "exit_partial") {
          if (!(typeof r.size_pct === "number" && r.size_pct > 0 && r.size_pct <= 100)) {
            errors.push({
              code: "INVALID_ACTION_PAYLOAD",
              path: `/asset_policies/${cls}/${side}/rules/${i}/size_pct`,
              message: "exit_partial requires size_pct in (0,100]"
            });
          }
        }
      });
    }
  }

  return { ok: errors.length === 0, errors, warnings };
}

export function validatePrelaunchReport(report: any): ValidationResult {
  const errors: Finding[] = [];
  const warnings: Finding[] = [];

  const okSchema = validatePrelaunchSchema(report);
  if (!okSchema) errors.push(...mapAjvErrors(validatePrelaunchSchema.errors));

  // Optional policy check: all error-severity checks must be pass/waived/not_applicable
  const checks = report?.checks ?? [];
  checks.forEach((c: any, i: number) => {
    if (c?.severity === "error" && c?.status === "fail") {
      errors.push({
        code: c?.code || "CHECK_FAILED",
        path: `/checks/${i}`,
        message: "prelaunch error-severity check failed"
      });
    }
  });

  return { ok: errors.length === 0, errors, warnings };
}
