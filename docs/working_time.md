# Statutory normal working hours, and why the per-hour ratio is not yet reported

The paper compares a *monthly* statutory wage with output per *person employed*.
A reviewer asked for the per-hour counterpart, which is the conceptually cleaner
object for a wage-productivity argument. This records how far that can currently
be built from primary law, because the answer decides whether the ratio can be
reported at all.

## Why hours matter here, and in which direction

Dividing a monthly wage by output per employed person is invariant to hours in
one narrow sense: deflating both sides by the same hours series cancels. That is
an assumption rather than a property, and the interesting failure is specific to
this country and this period. Normal weekly hours in Portuguese law fell over
the sample, so an unchanged monthly wage came to buy fewer hours. The hourly
wage floor therefore grew **faster** than the monthly figure the paper uses,
which means the per-worker ratio understates the recovery of the wage floor
rather than overstating it.

The conversion is not a convention we would have to choose. Article 271 of the
Código do Trabalho fixes it:

```text
hourly = (monthly * 12) / (52 * n)
```

where `n` is the normal weekly period. Note the 12: the fourteen annual payments
Portuguese law provides are separate entitlements and do not enter the hourly
base.

## What is established from the acts

**Lei n.º 21/96 of 23 July 1996** is retrievable at its ELI permalink and has
been read. Its article 1 does not do what secondary summaries say. It is not a
single step to forty hours:

> Os períodos normais de trabalho superiores a quarenta horas por semana são
> reduzidos nos seguintes termos: a) Na data da entrada em vigor da presente
> lei, são reduzidos de duas horas, até ao limite de quarenta horas;
> b) Decorrido um ano sobre a data de aplicação do disposto na alínea anterior,
> o remanescente é reduzido para quarenta horas.

So the reduction is phased: two hours at entry into force, the remainder a year
later. Its article 5 amends articles 10 and 12 of Decreto-Lei n.º 409/71, which
is the framework the limit actually lives in.

## What is not established, and why the ratio is therefore not reported

Three things block a series that would meet the standard the rest of this
project is held to.

- **Decreto-Lei n.º 409/71 of 27 September 1971**, which set the original limit,
  is not served as a PDF by the ELI system: the permalink returns an HTML shell.
  Acts predating 1974 are not in that collection. It has a detail page on
  `diariodarepublica.pt` but not one this pipeline can parse and checksum.
- **Lei n.º 2/91 of 17 January 1991**, understood to have set forty-four hours,
  resolves at its permalink but its PDF carries no extractable text layer. The
  retrieval succeeds and returns nothing, which is worse than failing.
- **Lei n.º 21/96 does not state its own entry into force.** The phase dates
  therefore depend on the general vacatio legis rule in force in 1996, which is
  a legal inference rather than something read from the act.

Two of the three legs of the series would be assertions. The series is not built
and the per-hour ratio is not reported, because a wage-productivity claim resting
on remembered dates is precisely the kind of claim this project exists to avoid.

## What would unblock it

Retrieving Decreto-Lei n.º 409/71 and Lei n.º 2/91 in a machine-readable form,
from the Assembleia da República's own archive or a text-layer copy of the
gazette, and establishing the 1996 phase dates from the general rule with a
citation. The conversion formula is already fixed by article 271, and the wage
series is already in place, so the remaining work is retrieval rather than
measurement.

An hours-worked productivity series reaching 1974 is a separate requirement and
has not been located either; AMECO's domestic-product chapter, which this
project already downloads, supplies output per person employed and not per hour.
