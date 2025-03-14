# Adding a New API Provider

This guide will walk you through the process of adding a new API provider to Llama Stack.


- Begin by reviewing the [core concepts](../concepts/index.md) of Llama Stack and choose the API your provider belongs to (Inference, Safety, VectorIO, etc.)
- Determine the provider type ({repopath}`Remote::llama_stack/providers/remote` or {repopath}`Inline::llama_stack/providers/inline`). Remote providers make requests to external services, while inline providers execute implementation locally.
- Add your provider to the appropriate {repopath}`Registry::llama_stack/providers/registry/`. Specify pip dependencies necessary.
- Update any distribution {repopath}`Templates::llama_stack/templates/` build.yaml and run.yaml files if they should include your provider by default. Run {repopath}`llama_stack/scripts/distro_codegen.py` if necessary. Note that `distro_codegen.py` will fail if the new provider causes any distribution template to attempt to import provider-specific dependencies. This usually means the distribution's `get_distribution_template()` code path should only import any necessary Config or model alias definitions from each provider and not the provider's actual implementation.


Here are some example PRs to help you get started:
   - [Grok Inference Implementation](https://github.com/meta-llama/llama-stack/pull/609)
   - [Nvidia Inference Implementation](https://github.com/meta-llama/llama-stack/pull/355)
   - [Model context protocol Tool Runtime](https://github.com/meta-llama/llama-stack/pull/665)


## Testing the Provider

### 1. Integration Testing
- Create integration tests that use real provider instances and configurations
- For remote services, test actual API interactions
- Avoid mocking at the provider level since adapter layers tend to be thin
- Reference examples in {repopath}`tests/api`

### 2. Unit Testing (Optional)
- Add unit tests for provider-specific functionality
- See examples in {repopath}`llama_stack/providers/tests/inference/test_text_inference.py`

### 3. End-to-End Testing
1. Start a Llama Stack server with your new provider
2. Test using client requests
3. Verify compatibility with existing client scripts in the [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main) repository
4. Document which scripts are compatible with your provider

## Submitting Your PR

1. Ensure all tests pass
2. Include a comprehensive test plan in your PR summary
3. Document any known limitations or considerations
4. Submit your pull request for review
