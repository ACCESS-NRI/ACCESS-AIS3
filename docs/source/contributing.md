# Contributing and maintaining the documentation

The documentation is manually authored. It explains design and workflow boundaries and
links to the implementation; it does not import workflow scripts or infer documentation
from their runtime objects.

## Change a workflow

When adding or changing a stage:

1. Update the relevant driver and launcher, keeping stage selection and output behavior clear.
2. Document the stage's input, transformation, output, restart/overwrite behavior, and
   validation state in the workflow guide or existing run guide.
3. Update the architecture page if the data flow or relationship between HO and SSA tracks
   changes.
4. Record scientific rationale and experimental evidence in the relevant worklog; keep
   old results clearly marked as superseded where needed.
5. Check that examples match the actual launcher, environment variables, and filenames.

Do not use autodocumentation that imports the drivers: their module-level setup is
environment-specific and can access data, alter process state, or initiate workflow
logic. A manually maintained script inventory is intentional.

## Build the site locally

Install the small documentation-only dependency set and build the HTML site from the
repository root:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs/source _build/html
```

The documentation build does not install pyISSM, ccdtools, or the model's scientific
dependencies. It must not submit cluster jobs or need access to model datasets.

## Publishing

The GitHub Actions workflow builds the Sphinx pages for pull requests and deploys them to
GitHub Pages after changes to the default branch. On initial setup, the repository's
**Settings → Pages** source must be set to **GitHub Actions**. Read the Docs configuration
is also provided for a future hosted Read the Docs project.

The existing development dashboard is published as a separate
[development record](development-record.md); its figures and worklogs are copied into
the Pages artifact so its relative links continue to work.