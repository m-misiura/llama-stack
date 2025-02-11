import yaml
import argparse


def create_deployment_yaml(index, image):
    base_name = f"regex-detector-{index}"

    deployment = {
        "kind": "Deployment",
        "apiVersion": "apps/v1",
        "metadata": {
            "name": base_name,
            "labels": {
                "app": "fmstack-nlp",
                "component": base_name,
                "deploy-name": base_name,
            },
        },
        "spec": {
            "replicas": 1,
            "selector": {
                "matchLabels": {
                    "app": "fmstack-nlp",
                    "component": base_name,
                    "deploy-name": base_name,
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": "fmstack-nlp",
                        "component": base_name,
                        "deploy-name": base_name,
                    }
                },
                "spec": {
                    "serviceAccountName": "user-one",
                    "containers": [
                        {
                            "name": base_name,
                            "image": image,
                            "command": ["/app/target/release/local-detectors"],
                            "env": [{"name": "HOST", "value": "0.0.0.0"}],
                            "ports": [
                                {
                                    "name": "http",
                                    "containerPort": 8080,
                                    "protocol": "TCP",
                                }
                            ],
                            "resources": {
                                "limits": {"cpu": "500m", "memory": "1Gi"},
                                "requests": {"cpu": "500m", "memory": "1Gi"},
                            },
                            "securityContext": {
                                "capabilities": {"drop": ["ALL"]},
                                "privileged": False,
                                "runAsNonRoot": True,
                                "readOnlyRootFilesystem": False,
                                "allowPrivilegeEscalation": False,
                                "seccompProfile": {"type": "RuntimeDefault"},
                            },
                            "imagePullPolicy": "Always",
                        }
                    ],
                },
            },
        },
    }

    service = {
        "kind": "Service",
        "apiVersion": "v1",
        "metadata": {
            "name": base_name,
            "labels": {"app": "fmstack-nlp", "component": base_name},
        },
        "spec": {
            "type": "ClusterIP",
            "ports": [
                {"name": "http", "protocol": "TCP", "port": 8080, "targetPort": 8080}
            ],
            "selector": {
                "app": "fmstack-nlp",
                "component": base_name,
                "deploy-name": base_name,
            },
        },
    }

    route = {
        "kind": "Route",
        "apiVersion": "route.openshift.io/v1",
        "metadata": {
            "name": f"{base_name}-route",
            "labels": {"app": "fmstack-nlp", "component": base_name},
            "annotations": {"openshift.io/host.generated": "true"},
        },
        "spec": {
            "to": {"kind": "Service", "name": base_name, "weight": 100},
            "port": {"targetPort": 8080},
        },
    }

    return [deployment, service, route]


def create_detector_config(index, hostname):
    return {
        f"regex-{index}": {
            "type": "text_contents",
            "service": {"hostname": hostname, "port": 80},
            "chunker_id": "whole_doc_chunker",
            "default_threshold": 0.5,
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate regex detector deployments and config"
    )
    parser.add_argument(
        "--detector-num",
        required=True,
        type=int,
        help="Number of regex detector deployments to generate",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="quay.io/rh-ee-mmisiura/local-detectors:pull_request",
        help="Container image to use for the deployments",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="apps.rosa.trustyai-mac.bd9q.p3.openshiftapps.com",
        help="Domain for the routes",
    )

    args = parser.parse_args()

    # Write deployments
    with open("regex_deployments.yaml", "w") as f:
        yaml_dumper = yaml.Dumper
        yaml_dumper.ignore_aliases = lambda self, data: True

        for i in range(args.detector_num):
            resources = create_deployment_yaml(i, args.image)
            for idx, resource in enumerate(resources):
                yaml.dump(
                    resource,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    Dumper=yaml_dumper,
                )
                if idx < len(resources) - 1 or i < args.detector_num - 1:
                    f.write("\n---\n")

    # Write detectors config
    detectors_config = {}
    for i in range(args.detector_num):
        hostname = f"regex-detector-{i}-route-test.{args.domain}"
        detectors_config.update(create_detector_config(i, hostname))

    with open("detectors_config.yaml", "w") as f:
        yaml.dump(
            {"detectors": detectors_config},
            f,
            default_flow_style=False,
            sort_keys=False,
        )


if __name__ == "__main__":
    main()
