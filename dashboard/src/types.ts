export interface CriterionScore {
  name: string
  score: number
  reasoning: string
}

export interface TestResult {
  test_id: string
  prompt: string
  model: string
  raw_output: string
  criterion_scores: CriterionScore[]
  weighted_score: number
  passed: boolean
  passing_threshold: number | null
  error: string | null
}

export interface RunSummary {
  run_id: string
  suite_name: string
  suite_version: string
  model: string
  judge_model: string
  status: string
  suite_path: string | null
  tags: string[]
  started_at: string
  completed_at: string | null
  total_tests: number
  passed_tests: number
  failed_tests: number
  errored_tests: number
  error_message: string | null
}

export interface SuiteRun extends RunSummary {
  results: TestResult[]
  concurrency: number
}

export interface TestDiff {
  test_id: string
  result_a: TestResult | null
  result_b: TestResult | null
  is_regression: boolean
  is_improvement: boolean
  score_delta: number | null
}
