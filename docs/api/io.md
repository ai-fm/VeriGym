# Import and Export

Functions for writing explicit environments to the model formats of external tools, and for loading models from such formats.
`.umb` and `.drn` files can be read by model checkers such as Storm and PRISM, while the `stormpy` exports return `stormpy` models in memory.
`gymnasium` environments and PRISM programs are loaded with `GenerativeEnv` (see [Environments](environments.md)), and policies computed by other tools with the classes in [Policies](policies.md).
For a walkthrough, see the [import and export guide](../guides/import_export.md).

## Export

```{eval-rst}
.. autofunction:: verigym.io.exporter.export_to_umb
```

```{eval-rst}
.. autofunction:: verigym.io.exporter.export_to_drn
```

```{eval-rst}
.. autofunction:: verigym.io.exporter.export_to_stormpy_mdp
```

```{eval-rst}
.. autofunction:: verigym.io.exporter.export_to_stormpy_imdp
```

## Import

```{eval-rst}
.. autofunction:: verigym.io.importer.load_from_umb
```

```{eval-rst}
.. autofunction:: verigym.io.importer.load_from_drn
```
