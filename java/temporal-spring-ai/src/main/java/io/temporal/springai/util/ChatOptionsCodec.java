package io.temporal.springai.util;

import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.util.JacksonUtils;
import tools.jackson.databind.annotation.JsonPOJOBuilder;
import tools.jackson.databind.cfg.MapperConfig;
import tools.jackson.databind.introspect.AnnotatedClass;
import tools.jackson.databind.introspect.JacksonAnnotationIntrospector;
import tools.jackson.databind.json.JsonMapper;

/** JSON mapper for immutable Spring AI options crossing an Activity boundary. */
public final class ChatOptionsCodec {
  private ChatOptionsCodec() {}

  /**
   * Uses each options type's public builder to reconstruct its concrete provider type. Tool
   * implementations stay in the workflow and cross the boundary as definitions only.
   */
  public static JsonMapper mapper() {
    return JacksonUtils.getDefaultJsonMapper()
        .rebuild()
        .annotationIntrospector(new OptionsIntrospector())
        .addMixIn(ToolCallingChatOptions.class, ToolOptionsMixin.class)
        .addMixIn(ToolCallingChatOptions.Builder.class, ToolOptionsMixin.class)
        .build();
  }

  @com.fasterxml.jackson.annotation.JsonIgnoreProperties(
      value = {"toolCallbacks", "toolNames", "toolContext"},
      ignoreUnknown = true)
  private abstract static class ToolOptionsMixin {}

  private static final class OptionsIntrospector extends JacksonAnnotationIntrospector {
    @Override
    public Class<?> findPOJOBuilder(MapperConfig<?> config, AnnotatedClass type) {
      if (ChatOptions.class.isAssignableFrom(type.getRawType())) {
        try {
          return type.getRawType().getMethod("builder").invoke(null).getClass();
        } catch (ReflectiveOperationException e) {
          // Custom bean-style options without a builder retain Jackson's normal handling.
        }
      }
      return super.findPOJOBuilder(config, type);
    }

    @Override
    public JsonPOJOBuilder.Value findPOJOBuilderConfig(
        MapperConfig<?> config, AnnotatedClass type) {
      if (ChatOptions.Builder.class.isAssignableFrom(type.getRawType())) {
        return new JsonPOJOBuilder.Value("build", "");
      }
      return super.findPOJOBuilderConfig(config, type);
    }
  }
}
