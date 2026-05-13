# zarrnii.plugins

Plugin API for segmentation and scaled-processing workflows, including hook
markers, hook specifications, plugin-manager helpers, and bundled plugins.

## Package exports

::: zarrnii.plugins
    handler: python
    options:
      show_root_heading: false
      show_object_full_path: true
      show_category_heading: true
      show_symbol_type_heading: false
      members_order: source
      merge_init_into_class: true
      show_docstring_description: true
      docstring_section_style: list
      show_source: true
      filters:
        - "!^_"  # hide private members

## Hook specifications

::: zarrnii.plugins.hookspecs
    handler: python
    options:
      show_root_heading: false
      show_object_full_path: true
      show_category_heading: true
      show_symbol_type_heading: false
      members_order: source
      merge_init_into_class: true
      show_docstring_description: true
      docstring_section_style: list
      show_source: true
      filters:
        - "!^_"  # hide private members

## Plugin manager

::: zarrnii.plugins.plugin_manager
    handler: python
    options:
      show_root_heading: false
      show_object_full_path: true
      show_category_heading: true
      show_symbol_type_heading: false
      members_order: source
      merge_init_into_class: true
      show_docstring_description: true
      docstring_section_style: list
      show_source: true
      filters:
        - "!^_"  # hide private members

## Built-in segmentation plugins

::: zarrnii.plugins.segmentation.local_otsu
    handler: python
    options:
      show_root_heading: false
      show_object_full_path: true
      show_category_heading: true
      show_symbol_type_heading: false
      members_order: source
      merge_init_into_class: true
      show_docstring_description: true
      docstring_section_style: list
      show_source: true
      filters:
        - "!^_"  # hide private members

::: zarrnii.plugins.segmentation.threshold
    handler: python
    options:
      show_root_heading: false
      show_object_full_path: true
      show_category_heading: true
      show_symbol_type_heading: false
      members_order: source
      merge_init_into_class: true
      show_docstring_description: true
      docstring_section_style: list
      show_source: true
      filters:
        - "!^_"  # hide private members

## Built-in scaled-processing plugins

::: zarrnii.plugins.scaled_processing.gaussian_biasfield
    handler: python
    options:
      show_root_heading: false
      show_object_full_path: true
      show_category_heading: true
      show_symbol_type_heading: false
      members_order: source
      merge_init_into_class: true
      show_docstring_description: true
      docstring_section_style: list
      show_source: true
      filters:
        - "!^_"  # hide private members

::: zarrnii.plugins.scaled_processing.n4_biasfield
    handler: python
    options:
      show_root_heading: false
      show_object_full_path: true
      show_category_heading: true
      show_symbol_type_heading: false
      members_order: source
      merge_init_into_class: true
      show_docstring_description: true
      docstring_section_style: list
      show_source: true
      filters:
        - "!^_"  # hide private members

::: zarrnii.plugins.scaled_processing.segmentation_cleaner
    handler: python
    options:
      show_root_heading: false
      show_object_full_path: true
      show_category_heading: true
      show_symbol_type_heading: false
      members_order: source
      merge_init_into_class: true
      show_docstring_description: true
      docstring_section_style: list
      show_source: true
      filters:
        - "!^_"  # hide private members
