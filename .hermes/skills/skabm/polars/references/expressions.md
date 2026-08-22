## Contents

- [String expressions](#string-expressions)
- [Temporal / datetime expressions](#temporal--datetime-expressions)
- [List expressions](#list-expressions)
- [Struct expressions](#struct-expressions)
- [Float / numeric expressions](#float--numeric-expressions)
- [Boolean / conditional expressions](#boolean--conditional-expressions)
- [Selector expressions](#selector-expressions)
- [Full fetch-map: 18 categories → live docs URLs](#full-fetch-map-18-categories--live-docs-urls)

Use this file when you are unsure which expression namespace a method lives in, or when you need the exact signature for a method outside the common `str`/`dt`/`list`/`struct` paths. Fetch the live docs URL for the category, then search the page for the method name.

---

## String expressions — `pl.col("x").str`

Methods on `Expr.str`. All operate on `String` columns. Verified on Polars 1.x.

- `.str.to_uppercase()` / `.str.to_lowercase()` — case conversion
- `.str.slice(offset, length)` — substring
- `.str.strip_chars()` / `.str.strip_chars_start()` / `.str.strip_chars_end()` — whitespace trimming
- `.str.split(by)` — split into `List[String]`
- `.str.split_exact(by, n)` — split into exactly `n+1` fields as struct
- `.str.concat(delimiter)` — join list elements
- `.str.contains(pattern, literal, strict)` — substring / regex match → Boolean
- `.str.ends_with(suffix)` / `.str.starts_with(prefix)` — prefix/suffix test
- `.str.extract(regex, group)` — extract capture group
- `.str.extract_all(regex)` — all matches as List
- `.str.replace(old, new, literal, n, return_count)` — replace
- `.str.replace_all(old, new, literal)` — replace all occurrences
- `.str.replace_nulls(replace)` — replace null with string
- `.str.lengths()` (deprecated → `.str.len_chars()`) — character count
- `.str.n_chars()` — character count (current name)
- `.str.json_decode(type)` — parse JSON string column into Struct/List
- `.str.json_path_match(path)` — JSONPath extraction
- `.str.parse_int(base, strict)` — string to integer
- `.str.to_date(format, strict, out_dtype)` — string to Date
- `.str.to_datetime(format, strict, exact, utc, tz, out_dtype)` — string to DateTime
- `.str.to_decimal(digits, strict)` — string to Decimal
- `.str.to_time(format, strict, out_dtype)` — string to Time
- `.str.zfill(width)` — zero-pad

---

## Temporal / datetime expressions — `pl.col("x").dt`

Methods on `Expr.dt`. Operate on `Date`, `Datetime`, `Duration`, `Time` columns.

- `.dt.year()` / `.dt.month()` / `.dt.day()` / `.dt.week()` / `.dt.weekday()` / `.dt.day_of_year()` / `.dt.quarter()` — date parts
- `.dt.hour()` / `.dt.minute()` / `.dt.second()` / `.dt.nanosecond()` — time parts
- `.dt.date()` — strip time, return Date
- `.dt.time()` — strip date, return Time
- `.dt.datetime()` — return DateTime (optionally with timezone)
- `.dt.strftime(format)` — format as string
- `.dt.to_string(format, exact)` — format as string (alternative)
- `.dt.to_datetime()` — convert Date → DateTime
- `.dt.to_date()` — convert DateTime → Date
- `.dt.to_time()` — convert DateTime → Time
- `.dt.days()` / `.dt.hours()` / `.dt.minutes()` / `.dt.seconds()` / `.dt.milliseconds()` / `.dt.microseconds()` / `.dt.nanoseconds()` — duration components
- `.dt.total_days()` / `.dt.total_hours()` / `.dt.total_minutes()` / `.dt.total_seconds()` / `.dt.total_milliseconds()` / `.dt.total_microseconds()` / `.dt.total_nanoseconds()` — duration as single unit
- `.dt.add_offset(seconds, minutes, hours, days, weeks, months, years, roll, time_unit)` — add offset
- `.dt.replace_time_zone(tz, ambiguous, nonexistent)` — attach/zone-shift
- `.dt.convert_time_zone(tz, ambiguous, nonexistent)` — convert timezone
- `.dt.combine(date, time)` — date + time → DateTime
- `.dt.truncate(interval, offset)` — truncate to interval boundary
- `.dt.offset_by(by, time_unit)` — shift by duration
- `.dt.epoch(tz, time_unit)` — epoch seconds/millis/micros/nanos
- `.dt.is_leap_year()` — Boolean
- `.dt.is_in_range(lower, upper, start_by, time_unit)` — range check
- `.dt.is_first_day_of_month()` / `.dt.is_last_day_of_month()` — month boundary
- `.dt.is_start_of_day()` / `.dt.is_end_of_day()` — day boundary
- `.dt.is_start_of_hour()` / `.dt.is_end_of_hour()` — hour boundary
- `.dt.is_start_of_minute()` / `.dt.is_end_of_minute()` — minute boundary
- `.dt.is_start_of_second()` / `.dt.is_end_of_second()` — second boundary
- `.dt.difference(other, time_unit)` — difference between two datetimes
- `.dt.last()` / `.dt.first()` — min/max in group context
- `.dt.cast(dtype)` — cast between temporal types

---

## List expressions — `pl.col("x").list`

Methods on `Expr.list`. Operate on `List` columns.

- `.list.len()` — length of each list
- `.list.min()` / `.list.max()` / `.list.sum()` / `.list.mean()` / `.list.median()` — aggregates over list elements
- `.list.first()` / `.list.last()` — first/last element
- `.list.get(index)` — element at index (negative indices wrap)
- `.list.head(n)` / `.list.tail(n)` — first/last n elements
- `.list.join(delimiter)` — join elements into string
- `.list.includes(item)` — Boolean: is item in list
- `.list.index_of(item, null_on_failure)` — first index of item
- `.list.slices(n, named)` — split into n-sized chunks
- `.list.get_chunks(n)` — chunk the list
- `.list.to_struct(fields, n_field_strategy)` — expand list into struct
- `.list.flatten()` — flatten list of lists
- `.list.sort(reverse)` — sort list elements
- `.list.unique()` — deduplicate list elements
- `.list.var()` / `.list.std()` — variance / stddev over elements
- `.list.quantile(quantile, interpolation)` — quantile over elements
- `.list.eval(expr, parallel)` — evaluate expression element-wise within each list
- `.list.shift(n, type_coercion)` — shift elements
- `.list.take(index)` — take elements by index list
- `.list.rechunk()` — rechunk list
- `.list.cast(dtype)` — cast list elements
- `.list.to_array(n, padding_value)` — convert list to fixed-size array
- `.list.fill_null(value)` — fill nulls in list

---

## Struct expressions — `pl.col("x").struct`

Methods on `Expr.struct`. Operate on `Struct` columns.

- `.struct.field(name)` — extract a field by name
- `.struct.rename_fields(names)` — rename fields
- `.struct.unnest()` — expand struct into separate columns
- `.struct.field("x")` is the common access pattern — use it when a column is a Struct and you need one member.

---

## Float / numeric expressions

Methods on numeric `Expr` (typically `Float32`/`Float64`/`Int*`).

- `.abs()` — absolute value
- `.ceil()` / `.floor()` / `.round()` / `.round(places)` — rounding
- `.sin()` / `.cos()` / `.tan()` / `.arcsin()` / `.arccos()` / `.arctan()` / `.arctan2(y)` — trig
- `.sinh()` / `.cosh()` / `.tanh()` / `.arcsinh()` / `.arccosh()` / `.arctanh()` — hyperbolic
- `.exp()` / `.exp2()` / `.log()` / `.log10()` / `.log2()` / `.ln()` — logs and exponents
- `.sqrt()` / `.cbrt()` — roots
- `.sign()` — sign of number
- `.is_finite()` / `.is_infinite()` / `.is_nan()` / `.is_not_nan()` / `.is_not_null()` / `.is_null()` — numeric predicates
- `.cast(dtype, strict)` — cast to another numeric type

---

## Boolean / conditional expressions

- `pl.when(cond).then(val).otherwise(val)` — ternary / case-when
- `.is_true()` / `.is_false()` — Boolean predicates
- `.all()` / `.any()` — aggregate over Boolean column
- `pl.col("x").eq_null_safe(other)` — null-safe equality
- `.ne_null_safe(other)` — null-safe inequality
- `.gt_null_safe(other)`, `.ge_null_safe(other)`, `.lt_null_safe(other)`, `.le_null_safe(other)` — null-safe comparison

---

## Selector expressions — `polars.selectors`

Import `from polars.selectors import ...`. Use selectors in `select`/`with_columns`/etc. to pick columns by dtype or name pattern instead of enumerating names.

- `cs.numeric()` — all numeric columns
- `cs.integer()` / `cs.floating()` / `cs.string()` / `cs.boolean()` / `cs.decimal()` / `cs.datetime()` / `cs.date()` / `cs.time()` / `cs.duration()` / `cs.list()` / `cs.struct()` / `cs.nested()` — dtype selectors
- `cs.by_dtype([...])` — columns matching given dtypes
- `cs.starts_with(prefix)` / `cs.ends_with(suffix)` / `cs.contains(substr)` — name pattern selectors
- `cs.numeric_null()` — numeric columns that contain nulls
- `cs.first()` / `cs.last()` — first/last column by position
- `cs.negate(selector)` / `~selector` — complement
- `cs.union(*selectors)` / `cs.intersection(*selectors)` — combine selectors

---

## Full fetch-map: 18 categories → live docs URLs

When a method is outside the namespaces above, fetch the live docs page for the category and search for the method name. The URLs are versioned — prefer the installed version's docs (check `pl.__version__`).

| Category | Docs URL pattern |
|---|---|
| Series | `https://docs.pola.rs/api/python/stable/reference/series/` |
| DataFrame | `https://docs.pola.rs/api/python/stable/reference/dataframe/` |
| LazyFrame | `https://docs.pola.rs/api/python/stable/reference/lazyframe/` |
| Expressions | `https://docs.pola.rs/api/python/stable/reference/expressions/` |
| GroupBy | `https://docs.pola.rs/api/python/stable/reference/groupby/` |
| Collations | `https://docs.pola.rs/api/python/stable/reference/collation/` |
| Selectors | `https://docs.pola.rs/api/python/stable/reference/selectors/` |
| SQL | `https://docs.pola.rs/api/python/stable/reference/sql/` |
| Testing | `https://docs.pola.rs/api/python/stable/reference/testing/` |
| Config | `https://docs.pola.rs/api/python/stable/reference/config/` |
| Functions | `https://docs.pola.rs/api/python/stable/reference/functions/` |
| NumPy conversions | `https://docs.pola.rs/api/python/stable/reference/numpy/` |
| PyTorch conversions | `https://docs.pola.rs/api/python/stable/reference/pytorch/` |
| Arrow conversions | `https://docs.pola.rs/api/python/stable/reference/arrow/` |
| Chunker | `https://docs.pola.rs/api/python/stable/reference/chunked/` |
| Indexed series | `https://docs.pola.rs/api/python/stable/reference/indexed_series/` |
| Utilities | `https://docs.pola.rs/api/python/stable/reference/utils/` |
| Options | `https://docs.pola.rs/api/python/stable/reference/options/` |
