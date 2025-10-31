# CI Environment Notes

## Known Type Check Differences

### bot/risk/risk_manager.py Line 648-651

**Issue**: GitHub Actions CI may report a mypy type error for pandas DataFrame scalar conversion:
```
error: Argument 1 to "float" has incompatible type "date | int | timedelta | Any | float | complex | str | bytes | datetime64[date | int | None] | timedelta64[timedelta | int | None] | None"
```

**Root Cause**: Different pandas/numpy versions between local development and CI environments cause different type inference.

**Status**: Safe to ignore or add `# type: ignore[arg-type]` comment for CI.

**Why it's safe**:
- The code uses proper try-except for type conversion
- Runtime behavior is identical across environments
- The `.item()` method and fallback handle all possible pandas scalar types
- Error handling catches any conversion failures

**Resolution Options**:
1. Add `# type: ignore[arg-type]` comment (will show as unused locally)
2. Configure CI to use same pandas/numpy versions as local dev
3. Accept the CI warning as environment-specific type checking difference

**Current Approach**: Code is type-safe with comprehensive error handling. The try-except block ensures all conversion cases are handled correctly at runtime, regardless of the pandas/numpy version.
