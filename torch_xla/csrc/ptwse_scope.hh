#pragma once

#include <string>

namespace ptwse {

class FrontendAttributePusher {
public:
    FrontendAttributePusher(const std::string& key, std::string value,
                            bool prefix_depth)
        : prefix_depth_(prefix_depth) {
        if (prefix_depth_) {
            const std::size_t current_depth =
                g_frontend_attribute_context.attributes.size();
            std::stringstream ss;
            ss << current_depth << "." << key;
            // Allow forced overwrite in case of duplicate
            key_ = ss.str();
        } else {
            key_ = key;
        }
        // Empty value means to erase from the map
        auto found = g_frontend_attribute_context.attributes.find(key_);
        if (found != g_frontend_attribute_context.attributes.end()) {
            previous_value_ = std::move(found->second);
            if (!value.empty()) {
                found->second = std::move(value);
            } else {
                g_frontend_attribute_context.attributes.erase(found);
            }
        } else {
            g_frontend_attribute_context.attributes.insert(
                std::make_pair(key_, std::move(value)));
        }
    }

    ~FrontendAttributePusher() {
        auto found = g_frontend_attribute_context.attributes.find(key_);
        // Entertain the possibility that it may have been removed within the scope
        // by something after the push operation
        if (found != g_frontend_attribute_context.attributes.end()) {
            if (previous_value_.empty()) {
                g_frontend_attribute_context.attributes.erase(found);
            } else {
                found->second = std::move(previous_value_);
            }
        } else if (!previous_value_.empty()) {
            g_frontend_attribute_context.attributes.insert(
                std::make_pair(key_, std::move(previous_value_)));
        }
    }

    static const std::unordered_map<std::string, std::string>&
    GetFrontendAttributes() {
        return g_frontend_attribute_context.attributes;
    }

    static void Reset() {
        return g_frontend_attribute_context.attributes.clear();
    }

    static void PythonAddFrontendAttribute(std::string key, std::string value) {
        g_frontend_attribute_context.attributes.emplace(std::move(key),
                                                        std::move(value));
    }

    static void PythonRemoveFrontendAttribute(const std::string& key) {
        g_frontend_attribute_context.attributes.erase(key);
    }

    static void PythonAddFrontendAttributes(
        const std::unordered_map<std::string, std::string>& attribute_map) {
        for (const auto& item : attribute_map) {
            PythonAddFrontendAttribute(item.first, item.second);
        }
    }

    static void PythonRemoveFrontendAttributes(
        const std::vector<std::string>& keys) {
        for (const auto& key : keys) {
            PythonRemoveFrontendAttribute(key);
        }
    }

    static const std::unordered_map<std::string, std::string>&
    GetPythonFrontendAttributes() {
        return g_frontend_attribute_context.attributes;
    }

private:
    struct FrontendAttributeContext {
        std::unordered_map<std::string, std::string> attributes;
        std::atomic<std::size_t> attribute_scope{0};
    };

    std::string key_;
    std::string previous_value_;
    const bool prefix_depth_;
    static thread_local FrontendAttributeContext g_frontend_attribute_context;

    friend inline std::string __partition_match_name(bool fwd);
};


class PartitionScope {
    static constexpr const char *MATCHED_OP = "MATCHED_OP";
public:
    std::string PartitionMatchName(bool fwd) {
        std::stringstream ss;
        ss << MATCHED_OP;
        if (fwd) {
            ss << ".FWD."
               << ++attribute_scope_;
        } else {
            ss << ".BWD."
               << ++attribute_scope_;
        }
        return ss.str();
    }
    inline std::string MakePartitionName(const std::string &function_name) {
        std::stringstream ss;
        ss << short_fn_name(function_name) << "<float16>(NAT,NAT)";
        return ss.str();
    }
    void Reset() {
        attribute_scope_ = 0;
    }
    static thread_local PartitionScope app;
private:

    static inline const char *prev_char(const char *original, const char *start, char c) {
        while (start > original && *start != c) {
            --start;
        }
        return start;
    }

    static inline std::string short_fn_name(const std::string &fn_name) {
        std::string result = fn_name;
        const char *start = fn_name.c_str();
        const char *s = strchr(start, '(');
        if (s && *s && s > start) {
            ++s;
            if (*s) {
                if (const char *s0 = prev_char(start, s - 1, ' ')) {
                    if (*s0 == ' ') {
                        ++s0;
                    }
                    const size_t sz = s - s0 + 1;
                    result = std::string(s0, sz);
                    std::replace(result.begin(), result.end(), ':', '_');
                }
            }
        }
        return std::move(result);
    }

    std::atomic<std::size_t> attribute_scope_{0};
};

#define PTWSE_INSTANTIATE_PARTITIONS() \
    thread_local ptwse::PartitionScope ptwse::PartitionScope::app; \
    thread_local ptwse::FrontendAttributePusher::FrontendAttributeContext \
        ptwse::FrontendAttributePusher::g_frontend_attribute_context;

#define PTWSE_ENABLE_PARITIONS_MACRO

#ifdef PTWSE_ENABLE_PARITIONS_MACRO
#define DECLARE_PARTITION()                    \
  ptwse::FrontendAttributePusher fattr(    \
      ptwse::PartitionScope::app.PartitionMatchName(true), \
      ptwse::PartitionScope::app.MakePartitionName(__FUNCTION__), \
      /*prefix_depth=*/true \
  )

#else
#define DECLARE_PARTITION() ((void)0)
#endif

/**
 * @brief Apply during lowering
 */
template<typename NODE_TYPE>
class FrontendAttributeSetter {
public:
    FrontendAttributeSetter(xla::XlaBuilder* builder,
                            const std::map<std::string, std::string>& attributes)
        : builder_(builder) {
        if (!attributes.empty()) {
            set_ = true;
            xla::FrontendAttributes frontend_attributes;
            frontend_attributes.CopyFrom(builder_->frontend_attributes());
            for (const auto& item : attributes) {
                frontend_attributes.mutable_map()->insert({item.first, item.second});
            }
            save_ = builder->SwapFrontendAttributes(frontend_attributes);
        }
    }
    FrontendAttributeSetter(xla::XlaBuilder* builder, const NODE_TYPE *node)
        : FrontendAttributeSetter(builder, node->GetFrontendAttributes()) {}
    ~FrontendAttributeSetter() {
        if (set_) {
            builder_->ClearOpMetadata();
            builder_->SetFrontendAttributes(save_);
        }
    }
    std::string Dump() {
        std::stringstream ss;
        for (const auto& item : builder_->frontend_attributes().map()) {
            ss << item.first << " -> " << item.second << ", ";
        }
        return ss.str();
    }

private:
    xla::XlaBuilder* builder_;
    xla::FrontendAttributes save_;
    bool set_ = false;
};


}  // end of ptwse namespace